"""Site-level alchemical mapping (`CoreMapping`) and its producers (`Mapper`).

An MSLD specification is, per site, a partition of particle indices into a shared
environment and a set of mutually-exclusive variant blocks. `CoreMapping` is the
canonical, force-field-agnostic representation of that partition for a single site:

    env_atoms    : indices shared by all variants (unscaled, instantiated once)
    blocks       : ordered {variant_name: indices}; one block == one lambda DOF
    attachments  : env->block boundary bonds. A list, since ring-closing residues
                   (e.g. proline) cross the boundary more than once
    align_xform  : per-variant rigid transform into the shared frame (optional)
    core_charges : fixed reference charge per env atom
    buffer_atoms : env atoms permitted a per-variant charge (subset of env_atoms)
    core_corresp : per-variant {variant core atom -> reference/env atom} correspondence
    atom_names   : hybrid atom name per combined index (post-rename)
    source_names : original atom name (in the variant it came from) per combined index

A `CoreMapping` is always tied to one concrete system (its atom indices), so it can only
exist once that system's indices exist. Two situations produce one:

  * SEPARATE per-variant structures: a `Mapper` discovers the correspondence (which atoms
    are shared) across the reference + variants and returns a `CoreMapping` blueprint. The
    hybrid structure is built later (assembly), not by the mapper.
  * PRE-COMBINED system: `ManualMapper` labels an already-assembled system (indices given
    explicitly). Useful for tests and as an escape hatch.

Everything downstream of `CoreMapping` (charge renormalization, `model_from_mappings`, the
build) is shared by both situations. A multi-site system is an ordered sequence of
`CoreMapping`s (one per site, 1..N); `HybridBlueprint` bundles them (+ n_atoms) and
`model_from_mappings` assembles them into a finalized `MSLDModel`.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ...model import MSLDModel


# ---------------------------------------------------------------------
# The shared data structure
# ---------------------------------------------------------------------
@dataclass
class CoreMapping:
    """Description of one site's alchemical partition.

    Particle indices are 0-based and refer to the target OpenMM System.

    env_atoms    : indices shared by all variants; instantiated once, never
                   lambda-scaled.
    blocks       : ordered {variant_name: indices}; each entry is one block == one
                   lambda DOF. Dict insertion order fixes block ordering within the site.
    attachments  : env->block boundary bonds as (env_atom, block_atom) pairs. A list,
                   since a variant may cross the boundary more than once (e.g. proline).
    align_xform  : {variant_name: rigid transform} placing each variant's coordinates
                   into the shared frame; empty when coordinates are already co-framed.
    core_charges : {env_atom: reference charge}.
    buffer_atoms : env atoms permitted a per-variant charge (subset of env_atoms).
    core_corresp : {variant_name: {variant_core_atom: reference_core_atom}} the shared-core
                   correspondence a Mapper discovered. Empty for a hand-labeled mapping.
    atom_names   : {combined_atom: hybrid name} after renaming (unique across the hybrid).
    source_names : {combined_atom: original name} in the variant the atom came from.
    variant_index_maps : {variant_name: {variant local atom -> combined atom}} for EVERY
                   atom of that variant (core atoms map onto the shared env index; block
                   atoms onto their appended index). What the assembler needs to graft.
    metadata     : free-form mapper annotations (e.g. {"backbone_mutation": True, ...}).
    """
    env_atoms: set[int]
    blocks: dict[str, set[int]]
    attachments: list[tuple[int, int]] = field(default_factory=list)
    align_xform: dict[str, "object"] = field(default_factory=dict)
    core_charges: dict[int, float] = field(default_factory=dict)
    buffer_atoms: set[int] = field(default_factory=set)
    core_corresp: dict[str, dict[int, int]] = field(default_factory=dict)
    atom_names: dict[int, str] = field(default_factory=dict)
    source_names: dict[int, str] = field(default_factory=dict)
    variant_index_maps: dict[str, dict[int, int]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # helpers
    @property
    def block_names(self) -> list[str]:
        """Variant names in order within the site."""
        return list(self.blocks.keys())

    def all_block_atoms(self) -> set[int]:
        """Every atom that belongs to some block at this site."""
        out: set[int] = set()
        for atoms in self.blocks.values():
            out |= atoms
        return out

    # validation
    def validate(self) -> "CoreMapping":
        """Check the mapping is internally consistent. Raises ValueError if not.

        Rejects malformed partitions before they reach the MSLDModel: empty
        blocks, a negative index, an index assigned to two blocks, env/block overlap,
        attachments referencing an unknown env or block atom, a buffer_atoms set not
        contained in env_atoms, and a core_corresp keyed by an unknown block name.
        Returns self, to allow fluent use.
        """
        if not self.blocks:
            raise ValueError("CoreMapping has no blocks (a site needs >= 1 variant)")

        # blocks non-empty, and no atom shared between two blocks
        owner: dict[int, str] = {}
        for name, atoms in self.blocks.items():
            if not atoms:
                raise ValueError(f"block '{name}' has no atoms")
            for a in atoms:
                if a < 0:
                    raise ValueError(f"block '{name}' has negative atom index {a}")
                if a in owner:
                    raise ValueError(
                        f"atom {a} is in two blocks: '{owner[a]}' and '{name}'")
                owner[a] = name

        # env and blocks must not overlap
        overlap = self.env_atoms & set(owner)
        if overlap:
            raise ValueError(
                f"atoms {sorted(overlap)} are in BOTH the environment and a block")

        # attachments must connect a real env atom to a real block atom
        block_atoms = set(owner)
        for e, b in self.attachments:
            if e not in self.env_atoms:
                raise ValueError(f"attachment env atom {e} is not in env_atoms")
            if b not in block_atoms:
                raise ValueError(f"attachment block atom {b} is not in any block")

        # buffer atoms must be part of the environment
        if not self.buffer_atoms <= self.env_atoms:
            bad = sorted(self.buffer_atoms - self.env_atoms)
            raise ValueError(f"buffer_atoms {bad} are not in env_atoms")

        # core_corresp / variant_index_maps, if present, must be keyed by known block names
        for label, dd in (("core_corresp", self.core_corresp),
                          ("variant_index_maps", self.variant_index_maps)):
            bad_names = sorted(set(dd) - set(self.blocks))
            if bad_names:
                raise ValueError(
                    f"{label} references unknown block name(s) {bad_names}")
        return self

    # serialization (JSON: object keys become strings, sets become sorted lists)
    def to_dict(self) -> dict:
        """Plain-JSON-able dict. Block insertion order (== lambda order) is preserved."""
        return {
            "env_atoms": sorted(self.env_atoms),
            "blocks": {name: sorted(atoms) for name, atoms in self.blocks.items()},
            "attachments": [list(p) for p in self.attachments],
            "align_xform": {name: (v.tolist() if hasattr(v, "tolist") else v)
                            for name, v in self.align_xform.items()},
            "core_charges": {str(a): q for a, q in self.core_charges.items()},
            "buffer_atoms": sorted(self.buffer_atoms),
            "core_corresp": {name: {str(k): v for k, v in d.items()}
                             for name, d in self.core_corresp.items()},
            "atom_names": {str(a): n for a, n in self.atom_names.items()},
            "source_names": {str(a): n for a, n in self.source_names.items()},
            "variant_index_maps": {name: {str(k): v for k, v in d.items()}
                                   for name, d in self.variant_index_maps.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoreMapping":
        return cls(
            env_atoms=set(d.get("env_atoms", [])),
            blocks={name: set(atoms) for name, atoms in d["blocks"].items()},
            attachments=[tuple(p) for p in d.get("attachments", [])],
            align_xform=dict(d.get("align_xform", {})),
            core_charges={int(a): q for a, q in d.get("core_charges", {}).items()},
            buffer_atoms=set(d.get("buffer_atoms", [])),
            core_corresp={name: {int(k): v for k, v in dd.items()}
                          for name, dd in d.get("core_corresp", {}).items()},
            atom_names={int(a): n for a, n in d.get("atom_names", {}).items()},
            source_names={int(a): n for a, n in d.get("source_names", {}).items()},
            variant_index_maps={name: {int(k): v for k, v in dd.items()}
                                for name, dd in d.get("variant_index_maps", {}).items()},
            metadata=dict(d.get("metadata", {})),
        )

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text: str) -> "CoreMapping":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------
# Multi-site container
# ---------------------------------------------------------------------
@dataclass
class HybridBlueprint:
    """The whole hybrid system's blueprint: n_atoms + one CoreMapping per site.

    sites[k] becomes MSLD site k+1 (site 0 / block 0 is the non-alchemical environment,
    never declared). `global_env` is everything not owned by a block. This is the object
    that serializes to one JSON and feeds `to_model()` / `model_from_mappings`.
    """
    n_atoms: int
    sites: list[CoreMapping]

    def validate(self) -> "HybridBlueprint":
        if self.n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {self.n_atoms}")
        if not self.sites:
            raise ValueError("blueprint needs >= 1 site")
        seen: dict[int, tuple[int, str]] = {}
        for si, m in enumerate(self.sites):
            m.validate()
            for name, atoms in m.blocks.items():
                for a in atoms:
                    if a >= self.n_atoms:
                        raise ValueError(
                            f"atom {a} (site {si}, block '{name}') >= n_atoms "
                            f"{self.n_atoms}")
                    if a in seen:
                        raise ValueError(
                            f"atom {a} is in two sites: {seen[a]} and {(si, name)}")
                    seen[a] = (si, name)
        return self

    @property
    def global_env(self) -> set[int]:
        """Every atom not owned by a block (env + rest of protein + solvent + ions)."""
        block_atoms: set[int] = set()
        for m in self.sites:
            block_atoms |= m.all_block_atoms()
        return set(range(self.n_atoms)) - block_atoms

    def to_model(self, **kw) -> MSLDModel:
        """Assemble into a finalized MSLDModel (see model_from_mappings)."""
        return model_from_mappings(self.n_atoms, self.sites, **kw)

    # serialization
    def to_dict(self) -> dict:
        return {"n_atoms": self.n_atoms, "sites": [m.to_dict() for m in self.sites]}

    @classmethod
    def from_dict(cls, d: dict) -> "HybridBlueprint":
        return cls(n_atoms=int(d["n_atoms"]),
                   sites=[CoreMapping.from_dict(s) for s in d["sites"]])

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text: str) -> "HybridBlueprint":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------
# Mapper base class
# ---------------------------------------------------------------------
class Mapper(ABC):
    """Base class for producing a `CoreMapping` (the blueprint), for one site.

    A Mapper answers "which atoms are shared env, which form each variant block" and
    returns a validated `CoreMapping`. It builds no structure. Two shapes exist:

      * DISCOVERY mappers (separate variants): they take a `reference` structure + the
        per-variant `variants` and discover the correspondence (e.g. by atom name for
        amino acids, by MCS for ligands). 
      * LABELING mappers (pre-combined system): indices are supplied explicitly
        (`ManualMapper`); `reference`/`variants` are ignored.

    Concrete mappers implement `get_mapping`. The signature is uniform so the assembly
    layer can call any mapper the same way.
    """

    @abstractmethod
    def get_mapping(self, reference=None, variants=None) -> CoreMapping:
        """Return a validated CoreMapping for one site."""
        raise NotImplementedError


# ---------------------------------------------------------------------
# Shared assembly: a per-variant core correspondence -> hybrid-local CoreMapping
# ---------------------------------------------------------------------
def unique_hybrid_name(base: str, used: set[str]) -> str:
    """First of base, base1, base2, ... not already in `used`."""
    if base not in used:
        return base
    i = 1
    while f"{base}{i}" in used:
        i += 1
    return f"{base}{i}"


def build_hybrid_local_mapping(blocks_struct, per_block_slot, core_slot_names, *,
                               metadata=None, rename=True) -> CoreMapping:
    """Assemble the hybrid-residue-local CoreMapping shared by every discovery mapper.

    blocks_struct   : ordered {variant_name: structure} (reference first if it competes). The
                      hybrid layout is env core (0..E-1) then each block's atoms appended.
    per_block_slot  : {variant_name: {local_pos: core_slot_index}} -- for each variant, which of
                      its atoms (by position) are core and which shared env slot they occupy.
                      A local position absent from this map is a block (appended) atom.
    core_slot_names : name for each env core slot (len E), used for atom_names of the env atoms.

    Whether the core was found by atom name (peptide) or by graph MCS (ligand), the assembly is
    identical: this is the one place that builds env_atoms / blocks / atom_names / source_names /
    core_corresp / variant_index_maps in the hybrid-local frame.
    """
    n_core = len(core_slot_names)
    env_atoms = set(range(n_core))
    atom_names = {i: core_slot_names[i] for i in range(n_core)}
    source_names = dict(atom_names)
    used = set(atom_names.values())
    hybrid = n_core

    blocks: dict[str, set[int]] = {}
    core_corresp: dict[str, dict[int, int]] = {}
    variant_index_maps: dict[str, dict[int, int]] = {}
    for name, st in blocks_struct.items():
        slotmap = per_block_slot.get(name, {})
        idx_map: dict[int, int] = {}
        corr: dict[int, int] = {}
        block_atoms: set[int] = set()
        for pos, a in enumerate(st.atoms):
            if pos in slotmap:
                h = slotmap[pos]                         # core atom -> its shared env index
                corr[pos] = h
            else:
                h = hybrid                               # block atom -> appended index
                hybrid += 1
                block_atoms.add(h)
                source_names[h] = a.name
                hn = unique_hybrid_name(a.name, used) if rename else a.name
                used.add(hn)
                atom_names[h] = hn
            idx_map[pos] = h
        blocks[name] = block_atoms
        core_corresp[name] = corr
        variant_index_maps[name] = idx_map

    return CoreMapping(
        env_atoms=env_atoms, blocks=blocks, atom_names=atom_names, source_names=source_names,
        core_corresp=core_corresp, variant_index_maps=variant_index_maps,
        metadata=dict(metadata or {}),
    ).validate()


# ---------------------------------------------------------------------
# The simplest mapper: atoms given explicitly
# ---------------------------------------------------------------------
@dataclass
class ManualMapper(Mapper):
    """Mapper for a predefined split (a labeling mapper).

    The environment/block assignment is supplied directly rather than inferred.
    Primary uses: deterministic test fixtures, and an override path when the automatic
    mappers are unsuitable. get_mapping() copies the inputs and validates them.
    """
    env_atoms: set[int]
    blocks: dict[str, set[int]]
    attachments: list[tuple[int, int]] = field(default_factory=list)
    core_charges: dict[int, float] = field(default_factory=dict)
    buffer_atoms: set[int] = field(default_factory=set)

    def get_mapping(self, reference=None, variants=None) -> CoreMapping:
        mapping = CoreMapping(
            env_atoms=set(self.env_atoms),
            blocks={name: set(atoms) for name, atoms in self.blocks.items()},
            attachments=list(self.attachments),
            core_charges=dict(self.core_charges),
            buffer_atoms=set(self.buffer_atoms),
        )
        return mapping.validate()


# ---------------------------------------------------------------------
# CoreMapping(s) -> MSLDModel
# ---------------------------------------------------------------------
def model_from_mappings(n_atoms: int, mappings, *, fnex: float = 5.5,
                        add_restraints: bool = False,
                        block_options: dict | None = None) -> MSLDModel:
    """Assemble an ordered sequence of per-site CoreMappings into a finalized MSLDModel.

    Site indexing: mappings[k] becomes site k+1. Site 0 / block 0 is the non-alchemical
    environment, it is never declared. Blocks are registered in site order, then variant
    order, since MSLDModel requires blocks added under consecutive, non-decreasing site
    indices.

    Only block atoms are assigned. Every particle not placed in a block stays in block 0
    (full strength, never lambda-scaled) - this includes the rest of the protein and all
    explicit solvent and ions.

    Per-block centroid restraints are added only when `add_restraints` is set (default off,
    ISS-001 / ADR-0033): they are NOT BLaDE's CATS (which groups equivalent atoms across
    substituents), and on a switched-off block (lambda ~ 0, its own LJ gone) they pull the side
    chain onto itself, driving its full-strength dU/dlambda to ~1e6 kJ/mol. A
    single-atom block is skipped: the restraint is realized as pairwise harmonic terms
    and is undefined for fewer than two atoms.

    `block_options` maps a variant name to extra Block keyword arguments (theta0,
    theta_mass, theta_friction, lambda_bias, fixed), unspecified blocks use defaults.

    Returns the model after finalize(), which additionally enforces MSLDModel's own
    invariants (consecutive site indices, >= 2 blocks per site unless a single fixed
    block, contiguous per-site block ranges).
    """
    # accept a single mapping or a list
    if isinstance(mappings, CoreMapping):
        mappings = [mappings]
    mappings = list(mappings)
    if not mappings:
        raise ValueError("need at least one CoreMapping (one site)")
    block_options = block_options or {}

    model = MSLDModel(n_atoms, fnex=fnex)

    # add all blocks of site 1, then site 2, ... (consecutive-site order required)
    for site_index, mapping in enumerate(mappings, start=1):
        mapping.validate()
        for name in mapping.block_names:
            atoms = sorted(mapping.blocks[name])
            model.add_block(site=site_index, atoms=atoms,
                            **block_options.get(name, {}))

    # optional per-block centroid restraints (skip single-atom blocks); see docstring
    if add_restraints:
        for mapping in mappings:
            for name in mapping.block_names:
                atoms = sorted(mapping.blocks[name])
                if len(atoms) >= 2:
                    model.add_atom_restraint(atoms)

    return model.finalize()
