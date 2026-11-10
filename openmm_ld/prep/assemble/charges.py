"""Partial-charge policy for the hybrid topology.

  * Environment atoms keep one fixed charge (they are shared and appear once).
  * Each block (variant) keeps its own charges, but we renormalize them by a small
    amount so that sum(env charges) + sum(block charges) == a whole-number target
    (the correct net charge of that variant's residue/molecule).
  * Net-charge guard: every variant at a site must reach the same integer target.
    If they differ, the total system charge would change with lambda, which breaks
    PME. Charge-changing perturbations need charge restraints that the engine does 
    not have yet.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..mapping.core import CoreMapping


@dataclass
class ChargeResult:
    """Outcome of renormalization.

    charges    : final per-atom charge (env atoms unchanged, block atoms adjusted).
    target     : variant name -> integer net charge it was forced to.
    raw_net    : variant name -> net charge BEFORE adjustment (env_local + raw block).
    correction : variant name -> total charge added to that block (target - raw_net).
    final_net  : variant name -> net charge AFTER adjustment (should equal target).
    """
    charges: dict[int, float]
    target: dict[str, int]
    raw_net: dict[str, float]
    correction: dict[str, float]
    final_net: dict[str, float]


# ---------------------------------------------------------------------
# Environment-charge averaging (needed for Amber)
# ---------------------------------------------------------------------
# The shared environment must carry one charge per atom.
# In CHARMM the backbone charges are standardized, so every variant already agrees.
# In Amber the charges are RESP-fit per residue, so a shared atom can carry a different
# charge in different variants. This is treated the following way: each atom charge becomes
# the per-atom average charge across the variants. The small residual is then absorbed by
# renormalize_charges.


@dataclass
class EnvChargeReport:
    """Outcome of averaging env charges across variants.

    averaged   : per-env-atom mean charge (same order as the input).
    differing  : list of (label, spread, avg, values) for atoms that were not equal.
    max_spread : the largest across-variant spread seen (in e).
    """
    averaged: list[float]
    differing: list[tuple]
    max_spread: float


def average_env_charges(per_variant, *, labels=None, equal_tol: float = 1e-6,
                        refuse_spread: float = 0.5, warn: bool = True) -> EnvChargeReport:
    """Average the environment charges across variants, per shared atom.

    per_variant : list over variants; each item is that variant's env charges in the
                  same order (position i is the same shared atom in every variant).
    labels      : optional names for the env atoms.

    Returns an EnvChargeReport. When any atom's charge differs across variants, 
    emits a warning saying we take the average. Raises if any atom's spread exceeds
    `refuse_spread`.
    """
    if not per_variant:
        raise ValueError("no variants given")
    n = len(per_variant[0])
    for k, pv in enumerate(per_variant):
        if len(pv) != n:
            raise ValueError(
                f"variant {k} has {len(pv)} env charges, expected {n} "
                f"(env atoms must correspond across variants)")

    averaged: list[float] = []
    differing: list[tuple] = []
    max_spread = 0.0
    for i in range(n):
        vals = [pv[i] for pv in per_variant]
        spread = max(vals) - min(vals)
        avg = sum(vals) / len(vals)
        averaged.append(avg)
        max_spread = max(max_spread, spread)
        if spread > equal_tol:
            label = labels[i] if labels is not None else i
            if spread > refuse_spread:
                raise ValueError(
                    f"env atom {label} charge varies by {spread:.3f} e across variants "
                    f"(> {refuse_spread}); it is not really a shared atom. This looks "
                    f"like a backbone mutation, not a side-chain site.")
            differing.append((label, spread, avg, list(vals)))

    if differing and warn:
        names = ", ".join(str(d[0]) for d in differing[:6])
        more = "" if len(differing) <= 6 else f" (+{len(differing) - 6} more)"
        warnings.warn(
            f"env charges differ across variants for {len(differing)} atom(s) "
            f"[{names}{more}], max spread {max_spread:.3f} e; taking the per-atom "
            f"average. Amber charges are residue-specific (RESP-fit), so this is "
            f"expected.", stacklevel=2)
    return EnvChargeReport(averaged=averaged, differing=differing, max_spread=max_spread)


def _require_charges(mapping: CoreMapping, charges: dict[int, float]) -> None:
    """Every env atom and every block atom must have a charge."""
    needed = set(mapping.env_atoms) | mapping.all_block_atoms()
    missing = sorted(needed - set(charges))
    if missing:
        raise ValueError(f"charges missing for atoms {missing}")


def variant_net_charges(mapping: CoreMapping,
                        charges: dict[int, float]) -> dict[str, float]:
    """Net charge of each variant's end state = sum(env charges) + sum(block charges).

    The environment sum is the same for every variant (env is shared), so the only
    thing that changes between variants is their own block.
    """
    _require_charges(mapping, charges)
    env_sum = sum(charges[a] for a in mapping.env_atoms)
    out: dict[str, float] = {}
    for name, atoms in mapping.blocks.items():
        out[name] = env_sum + sum(charges[a] for a in atoms)
    return out


def _resolve_targets(nets: dict[str, float], targets: dict[str, int] | None,
                     tol: float) -> dict[str, int]:
    """Decide the integer target for each variant.

    If `targets` is given, use it. Otherwise auto-detect: a correctly parametrized
    residue sums to an integer already, so we round - but only if we are close to an
    integer. If a raw net is far from any integer and no target was given, we refuse
    (it usually means missing atoms or wrong charges), and ask for explicit targets.
    """
    resolved: dict[str, int] = {}
    for name, net in nets.items():
        if targets is not None and name in targets:
            resolved[name] = int(targets[name])
            continue
        nearest = round(net)
        if abs(net - nearest) > tol:
            raise ValueError(
                f"variant '{name}' raw net charge {net:.4f} is far from an integer "
                f"(nearest {nearest}, tol {tol}). Pass explicit `targets` or check the "
                f"charges -- auto-rounding could pick the wrong integer.")
        resolved[name] = nearest
    return resolved


def assert_iso_charge(mapping: CoreMapping, charges: dict[int, float], *,
                      targets: dict[str, int] | None = None,
                      tol: float = 0.25) -> int:
    """Net-charge guard. Raises unless all variants share one integer net charge.

    Returns that shared integer. Use this to fail fast before building anything.
    """
    nets = variant_net_charges(mapping, charges)
    resolved = _resolve_targets(nets, targets, tol)
    distinct = set(resolved.values())
    if len(distinct) != 1:
        detail = ", ".join(f"{k}={v:+d}" for k, v in resolved.items())
        raise ValueError(
            f"variants at this site have different net charges ({detail}). MSLD needs "
            f"the same net charge on every variant; charge-changing perturbations need "
            f"charge restraints, which are not implemented (see vault FEAT-037).")
    return distinct.pop()


def renormalize_charges(mapping: CoreMapping, charges: dict[int, float], *,
                        targets: dict[str, int] | None = None,
                        tol: float = 0.25,
                        method: str = "spread",
                        allow_buffer: bool = False) -> ChargeResult:
    """Adjust each block's charges so its end state hits a clean integer net charge.

    Environment charges are left untouched. The small correction (target - raw_net) is
    distributed over the block's atoms:
      - method "spread"       : equal share to every block atom (simplest, default).
      - method "proportional" : share weighted by |charge| (leaves near-zero atoms
                                 almost untouched).

    Enforces the net-charge guard (all variants same integer). Returns a ChargeResult.
    `allow_buffer` permits mapping.buffer_atoms (boundary env atoms handled by
    BufferAwareRenormalizer); otherwise their presence raises.
    """
    _require_charges(mapping, charges)
    if mapping.buffer_atoms and not allow_buffer:
        raise NotImplementedError(
            "buffer_atoms (per-variant charge on shared atoms) needs the buffer policy; use "
            "BufferAwareRenormalizer, or leave buffer_atoms empty.")
    if method not in ("spread", "proportional"):
        raise ValueError(f"unknown method '{method}' (use 'spread' or 'proportional')")

    raw_net = variant_net_charges(mapping, charges)
    target = _resolve_targets(raw_net, targets, tol)

    # net-charge guard: one shared integer for the whole site
    distinct = set(target.values())
    if len(distinct) != 1:
        detail = ", ".join(f"{k}={v:+d}" for k, v in target.items())
        raise ValueError(
            f"variants at this site have different net charges ({detail}). MSLD needs "
            f"the same net charge on every variant; charge-changing perturbations need "
            f"charge restraints, which are not implemented (see vault FEAT-037).")

    new_charges = dict(charges)          # copy; env atoms stay as-is
    correction: dict[str, float] = {}
    final_net: dict[str, float] = {}

    for name, atoms in mapping.blocks.items():
        atoms = sorted(atoms)
        corr = target[name] - raw_net[name]
        correction[name] = corr

        if method == "spread":
            weights = [1.0] * len(atoms)
        else:  # proportional to |charge|
            weights = [abs(charges[a]) for a in atoms]
        wsum = sum(weights)
        if wsum == 0.0:                  # all-zero charges -> fall back to equal spread
            weights = [1.0] * len(atoms)
            wsum = float(len(atoms))

        for a, w in zip(atoms, weights):
            new_charges[a] = charges[a] + corr * (w / wsum)

        env_sum = sum(new_charges[e] for e in mapping.env_atoms)
        final_net[name] = env_sum + sum(new_charges[a] for a in atoms)

    return ChargeResult(charges=new_charges, target=target, raw_net=raw_net,
                        correction=correction, final_net=final_net)


# ---------------------------------------------------------------------
# ChargeRenormalization — the class facade (Phase E)
# ---------------------------------------------------------------------
# A thin, swappable wrapper over the validated numerics above that operates IN PLACE on the
# hybrid structure. `apply` does two things:
#   1. env averaging (optional): the shared env atoms differ across variants in Amber (RESP
#      is per-residue), so each shared atom gets the per-atom mean. Needs the variant
#      structures (the hybrid instantiates each shared atom once, keeping the reference's
#      value), which it reaches via mapping.variant_index_maps + mapping.metadata["reference"].
#   2. block renormalization: nudge each block so its end-state net charge is a clean integer
#      (method "spread" or "proportional"), and enforce the net-charge guard.
#
# Duck-typed on the structure: reads/writes structure.atoms[i].charge / .name only (works on a
# ParmEd Structure or any equivalent), so this stays free of a parmed import.


@dataclass
class RenormReport:
    """Outcome of ChargeRenormalization.apply."""
    charges: dict                       # final per-atom charges written to the structure
    block: ChargeResult                 # block-renormalization detail
    env: "EnvChargeReport | None" = None  # env-averaging detail (None if not averaged)


@dataclass
class ChargeRenormalization:
    """Make the hybrid's charges consistent: average the env, renormalize each block.

    method       : "spread" (equal share) or "proportional" (weighted by |q|).
    average_env  : average the shared env charges across variants (needs `variants`).
    targets/tol  : passed through to the integer-target resolution + net-charge guard.
    """
    method: str = "spread"
    average_env: bool = True
    targets: dict | None = None
    tol: float = 0.25
    allow_buffer: bool = False

    def apply(self, structure, mapping: CoreMapping, variants=None) -> RenormReport:
        """Renormalize `structure`'s charges in place. Returns a RenormReport."""
        env_report = None
        if self.average_env and variants:
            env_report = self._average_env(structure, mapping, variants)

        atoms = sorted(set(mapping.env_atoms) | mapping.all_block_atoms())
        charges = {i: float(structure.atoms[i].charge) for i in atoms}
        result = renormalize_charges(mapping, charges, targets=self.targets,
                                     tol=self.tol, method=self.method,
                                     allow_buffer=self.allow_buffer)
        for i, q in result.charges.items():
            structure.atoms[i].charge = q
        return RenormReport(charges=result.charges, block=result, env=env_report)

    def _average_env(self, structure, mapping: CoreMapping, variants) -> EnvChargeReport:
        ref = mapping.metadata.get("reference")
        # buffer atoms are boundary env atoms whose charge is allowed to depend on the variant,
        # so they are NOT averaged (they keep the reference value here; full per-variant buffer
        # charges need structural duplication -- see BufferAwareRenormalizer).
        env_sorted = [g for g in sorted(mapping.env_atoms) if g not in mapping.buffer_atoms]
        labels = [structure.atoms[g].name for g in env_sorted]
        per_variant = []
        for vname in mapping.blocks:                # every competing variant (incl. reference)
            inv = {g: loc for loc, g in mapping.variant_index_maps.get(vname, {}).items()}
            row = []
            for g in env_sorted:
                # the reference variant lives in the hybrid; the others in `variants`
                if vname != ref and vname in variants and g in inv:
                    row.append(float(variants[vname].atoms[inv[g]].charge))
                else:
                    row.append(float(structure.atoms[g].charge))
            per_variant.append(row)
        report = average_env_charges(per_variant, labels=labels)
        for g, q in zip(env_sorted, report.averaged):
            structure.atoms[g].charge = q
        return report


@dataclass
class SpreadRenormalizer(ChargeRenormalization):
    """Distribute each block's integer-net correction EQUALLY over its atoms (default)."""
    method: str = "spread"


@dataclass
class ProportionalRenormalizer(ChargeRenormalization):
    """Distribute the correction weighted by |charge| (near-zero atoms stay near zero)."""
    method: str = "proportional"


@dataclass
class BufferAwareRenormalizer(ChargeRenormalization):
    """Renormalizer that TOLERATES buffer atoms (boundary env atoms, e.g. the ligand/MCS path).

    Buffer atoms are excluded from env averaging (they keep the reference charge) and are allowed
    through the block renormalization. NOTE: giving each variant its OWN charge on a buffer atom
    additionally needs the buffer atom DUPLICATED into each block at the structure level (a Phase
    D operation); this class handles the charge policy for the single-instance case and unblocks
    the net-charge accounting. Full per-variant buffer charges remain a deferred structural item.
    """
    allow_buffer: bool = True
