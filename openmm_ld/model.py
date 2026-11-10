"""MSLD data model (ported from OpenMM-MSLD msld/model.py).

Python code mirroring BLaDE's src/msld/msld.h / msld.cu
(no OpenMM objects)

Concepts
--------
site   : a set of overlapping alternative substituents.
         Site 0 = environment (always fully present).
block  : one substituent = a set of atoms + one theta DOF. Block 0 = environment.
lambda : lambda_b in [0,1]; per site sum(lambda) == 1.

The energy scaling by block lambdas is defined by the following rules:
  - classify_bonded : product of the distinct block lambdas
                      whose atoms appear in a bonded term  (bilinear scaling).
  - classify_pair   : nonbonded pair rule
                        same block             -> single lambda_b
                        same site, diff block  -> EXCLUDED (scale 0; mutually exclusive substituents)
                        different sites / env  -> product lambda_i * lambda_j
                        both environment       -> unscaled
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

# theta units (ADR-0038). theta is in radians and energies in kJ/mol, so the theta mass carries
# amu*nm^2. BLaDE (AKMA build, the one CHARMM and ALF use) takes its theta mass in amu*A^2:
# its default 5 equals 0.05 here. The NMPS build already uses nm, so its mass maps 1:1.
_BLADE_TO_NM2 = {"AKMA": 0.01, "NMPS": 1.0}


def theta_mass_from_blade(m: float, units: str = "AKMA") -> float:
    """Convert a BLaDE theta mass to ours (amu*nm^2). units: 'AKMA' (CHARMM/ALF, A^2) or 'NMPS'."""
    try:
        return float(m) * _BLADE_TO_NM2[units.upper()]
    except KeyError:
        raise ValueError(f"unknown BLaDE unit system {units!r} (AKMA or NMPS)") from None


@dataclass
class Block:
    """One substituent (or the environment, index 0)."""
    index: int                       # 0 = environment
    site: int                        # 0 = environment site
    atoms: list[int] = field(default_factory=list)
    theta0: float = 0.0              # initial theta
    theta_velocity0: float = 0.0
    theta_mass: float = 0.05         # amu*nm^2 (theta in rad); = BLaDE's 5 amu*A^2 (AKMA build)
    theta_friction: float = 1.0      # ps^-1 (per-block Langevin friction), as BLaDE
    lambda_bias: float = 0.0         # fixed linear bias coefficient on lambda
    fixed: bool = False              # freeze THETA. lambda is fixed only if every block of the
                                     # site is fixed (the softmax renormalizes over the site)


@dataclass
class VariableBias:
    """ALF-style coupling bias between two blocks (BLaDE LDBV)."""
    i: int
    j: int
    type: int
    l0: float
    k: float
    n: int


class MSLDModel:
    """The full MSLD specification for one system."""

    def __init__(self, n_atoms: int, fnex: float = 5.5):
        self.n_atoms = int(n_atoms)
        self.fnex = float(fnex)
        # Block 0 = environment; owns no explicit atoms (every unassigned atom is env).
        self.blocks: list[Block] = [Block(index=0, site=0)]
        self.atom_block: list[int] = [0] * self.n_atoms
        self.variable_biases: list[VariableBias] = []
        # which bonded term types get lambda-scaled (BLaDE "removescaling").
        # BLaDE runs usually turn bond+angle scaling OFF.
        self.scale_bond = True
        self.scale_angle = True
        self.scale_torsion = True
        # ALF: when True, scale_system builds a TUNABLE fixed-bias term for every alchemical block
        # (even at coefficient 0), so alf.flatten can retune the biases live (see msld/alf.py).
        self.bias_flattening = False
        # atom restraints (CATS): keep each listed group near its own centroid.
        self.atom_restraints: list[list[int]] = []
        self.k_restraint = 24769.0        # kJ/mol/nm^2 (BLaDE default 59.2 kcal/mol/A^2)
        # soft-core for overlap-prone nonbonded pairs (block<->env, block<->block across sites):
        # Beutler alpha for LJ and Coulomb, and the fixed Coulomb soft-core length (nm).
        # alpha = 0 restores plain linear lambda scaling.
        self.softcore_alpha = 0.5
        self.softcore_sigma_q = 0.3
        self._finalized = False

    def add_block(self, site: int, atoms, **kw) -> int:
        """Add a substituent block at site covering atoms. Returns its index."""
        if self._finalized:
            raise RuntimeError("model already finalized")
        if site < 1:
            raise ValueError("substituent blocks must have site >= 1 (0 is environment)")
        idx = len(self.blocks)
        atoms = list(atoms)
        for a in atoms:
            if not (0 <= a < self.n_atoms):
                raise ValueError(f"atom {a} out of range [0,{self.n_atoms})")
            if self.atom_block[a] != 0:
                raise ValueError(f"atom {a} already in block {self.atom_block[a]}")
            self.atom_block[a] = idx
        self.blocks.append(Block(index=idx, site=site, atoms=atoms, **kw))
        return idx

    def add_variable_bias(self, i, j, type, l0, k, n):
        self.variable_biases.append(VariableBias(i, j, type, l0, k, n))

    def add_atom_restraint(self, atoms):
        """Restrain a group of atoms (e.g. one substituent) near its centroid."""
        self.atom_restraints.append(list(atoms))

    def finalize(self):
        """Validate invariants and precompute site bookkeeping (BLaDE initialize)."""
        # blocks must be ordered by consecutive sites (BLaDE requirement)
        for b in range(1, len(self.blocks)):
            s, sp = self.blocks[b].site, self.blocks[b - 1].site
            if s not in (sp, sp + 1):
                raise ValueError(
                    f"blocks must be ordered by consecutive sites; block {b} "
                    f"(site {s}) out of order after block {b-1} (site {sp})")
        self.n_sites = max(bl.site for bl in self.blocks) + 1
        # blocks per site + contiguous bounds
        self.blocks_per_site = [0] * self.n_sites
        for bl in self.blocks:
            self.blocks_per_site[bl.site] += 1
        if self.blocks_per_site[0] != 1:
            raise ValueError("site 0 (environment) must contain exactly one block")
        self.site_bound = [0] * (self.n_sites + 1)
        for s in range(self.n_sites):
            n = self.blocks_per_site[s]
            if s >= 1 and n < 2 and not (n == 1 and self.blocks[self.site_bound[s]].fixed):
                raise ValueError(
                    f"site {s} needs >=2 blocks (or a single fixed block); found {n}")
            self.site_bound[s + 1] = self.site_bound[s] + n
        self._finalized = True
        return self

    def blocks_in_site(self, s: int) -> list[int]:
        return list(range(self.site_bound[s], self.site_bound[s + 1]))

    def classify_bonded(self, atoms, max_blocks: int = 2) -> list[int]:
        """Distinct non-environment blocks whose lambda scale a bonded term.

        Returns [] (env-only), [b] (single factor), or [bi, bj] (product), etc.
        Raises if the term spans two different blocks of the same site (illegal).
        """
        blocks = sorted({self.atom_block[a] for a in atoms
                         if self.atom_block[a] != 0}, reverse=True)
        seen: dict[int, int] = {}
        for b in blocks:
            s = self.blocks[b].site
            if s in seen and seen[s] != b:
                raise ValueError(
                    f"bonded term spans blocks {seen[s]} and {b} in the same "
                    f"site {s} (illegal MSLD scaling)")
            seen[s] = b
        if len(blocks) > max_blocks:
            raise ValueError(
                f"bonded term needs {len(blocks)} lambda factors > max {max_blocks}")
        return blocks

    def classify_pair(self, i: int, j: int):
        """Nonbonded pair scaling rule. Returns one of:
            ("env",)            both environment -> unscaled
            ("single", b)       one lambda factor
            ("product", bi, bj) product of two lambdas (different sites)
            ("exclude",)        same site, different block -> interaction removed
        """
        bi, bj = self.atom_block[i], self.atom_block[j]
        if bi == 0 and bj == 0:
            return ("env",)
        if bi == bj:
            return ("single", bi)
        if bi != 0 and bj != 0 and self.blocks[bi].site == self.blocks[bj].site:
            return ("exclude",)
        if bi == 0:
            return ("single", bj)
        if bj == 0:
            return ("single", bi)
        return ("product", bi, bj)

    def lambda_from_theta(self, theta) -> list[float]:
        """Reference (CPU) softmax projection, per site. For testing vs BLaDE."""
        lam = [0.0] * len(self.blocks)
        for s in range(self.n_sites):
            js = self.blocks_in_site(s)
            if s == 0:
                lam[js[0]] = 1.0
                continue
            w = [math.exp(self.fnex * math.sin(theta[b])) for b in js]
            z = sum(w)
            for b, wb in zip(js, w):
                lam[b] = wb / z
        return lam

    def scale_system(self, system, check_unhandled: bool = True) -> dict:
        """Apply MSLD block-scaling to `system` in place; return the info dict.

        The method form of the engine entry point (see builder.scale_msld_system). The model
        must be finalized. Imported lazily so importing the data model does not pull in
        OpenMM/forces.
        """
        from .builder import scale_msld_system
        return scale_msld_system(system, self, check_unhandled=check_unhandled)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def summary(self) -> str:
        lines = [f"MSLDModel: {self.n_atoms} atoms, {self.n_blocks} blocks, "
                 f"fnex={self.fnex}"]
        if self._finalized:
            for s in range(self.n_sites):
                bs = self.blocks_in_site(s)
                lines.append(f"  site {s}: blocks {bs}")
        return "\n".join(lines)


def model_from_mapping(mapping, *, fnex: float = 5.5, block_options: dict | None = None) -> MSLDModel:
    """A finalized MSLDModel from a hybrid mapping dict, as saved in <prefix>_mapping.json:
    {"n_atoms": N, "sites": [{"blocks": {name: [atoms]}, ...}, ...]}. mappings[k] becomes site k + 1,
    blocks in the file's order. Every other atom stays in the environment. `block_options` maps a
    variant name to extra Block keyword arguments."""
    block_options = block_options or {}
    model = MSLDModel(int(mapping["n_atoms"]), fnex=fnex)
    for index, site in enumerate(mapping["sites"], start=1):
        for name, atoms in site["blocks"].items():
            model.add_block(site=index, atoms=sorted(atoms), **block_options.get(name, {}))
    return model.finalize()
