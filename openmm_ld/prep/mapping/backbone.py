"""Amino-acid mapper: discover the shared core across residue-level variants.

`AminoAcidBackboneMapper` implements the single rule:

    env = the maximal common core matched by atom name and atom type.

Env atoms are unscaled, so they must carry one Lennard-Jones type -> the type must agree
across every variant. Charges may still differ (they are averaged later). Everything an atom
does not share (a different name, or the same name with a different type) becomes part of that
variant's block. When a backbone heavy atom (N, CA, C, O) falls into the blocks, the result is
flagged as a backbone mutation (metadata["backbone_mutation"]).

    Amber ff14SB : backbone types are identical across ALA/GLY/PHE/PRO/TYR and there is no
                   CMAP -> the whole backbone stays shared; Pro/Gly are side-chain-scope.
    CHARMM36 / Amber ff19SB : backbone types are residue-specific (Gly CA=CT2, Pro N=N,
                   CA=CP1) and CMAP is present -> those atoms differ in type, fall into the
                   blocks = a backbone mutation, which the engine's CMAP guard then refuses.

The mapper reads atom .name / .idx and the LJ parameters (.atomic_number / .sigma / .epsilon,
falling back to .type when LJ is absent), so it is force-field agnostic and needs no ParmEd
import; it works on any structure whose `.atoms` expose those attributes.

Output frame: the emitted `CoreMapping` is in a hybrid-residue-local frame it constructs --
core atoms first (0..E-1, in reference order), then each block's unique atoms appended. The
assembler (later phase) offsets these into the full solvated system. `variant_index_maps`
gives, per variant, the {local atom -> hybrid atom} map the assembler needs to graft.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import CoreMapping, Mapper

# Heavy backbone atoms whose presence in a block marks a BACKBONE mutation.
BACKBONE_HEAVY = ("N", "CA", "C", "O")


def _lj_key(atom):
    """A stable 'is this the same atom kind?' key: element + LJ (sigma, epsilon), rounded.

    Env atoms are unscaled, so a shared-core atom must carry ONE Lennard-Jones parameter set --
    LJ equality (not the atom-TYPE string) is the real criterion. Type-name strings are synthesized
    per structure by ParmEd's OpenMM->ParmEd hop, so they differ for physically identical atoms in
    separately-ingested variants. Falls back to the type string only when LJ is UNASSIGNED (a bare
    atom that exposes just name/type -- e.g. a synthetic test structure); a real atom_type with a
    legitimately zero LJ (e.g. a hydroxyl H) still compares by LJ."""
    at = getattr(atom, "atom_type", None)
    # ParmEd's placeholder for "no LJ assigned" is truthy and reports sigma=eps=0, so detect it by
    # class name (import-free) rather than by truthiness or a zero LJ.
    if at is not None and type(at).__name__ != "_UnassignedAtomType":
        try:
            return ("lj", int(atom.atomic_number),
                    round(float(atom.sigma), 4), round(float(atom.epsilon), 5))
        except Exception:
            pass
    return ("type", getattr(atom, "type", None))


@dataclass
class AminoAcidBackboneMapper(Mapper):
    """Name+type common-core mapper for amino-acid variants (one site).

    reference_name    : block label for the wild-type (reference) residue.
    include_reference : if True (default, per [D3]) the reference residue competes as a
                        block too; it always defines the env frame regardless.
    backbone_atoms    : heavy-atom names used only to decide the backbone_mutation flag.
    rename_blocks     : make appended block atom names unique across the hybrid residue.
    """
    reference_name: str = "WT"
    include_reference: bool = True
    backbone_atoms: tuple = BACKBONE_HEAVY
    rename_blocks: bool = True

    def get_mapping(self, reference=None, variants=None) -> CoreMapping:
        if reference is None:
            raise ValueError("AminoAcidBackboneMapper needs a reference structure")
        variants = dict(variants or {})

        # competing blocks, in order: reference (optional) first, then the variants
        blocks_struct: dict[str, object] = {}
        if self.include_reference:
            blocks_struct[self.reference_name] = reference
        for name, st in variants.items():
            blocks_struct[name] = st
        if len(blocks_struct) < 2:
            raise ValueError(
                "a site needs >= 2 competing variants (pass more variants, or keep "
                "include_reference=True)")

        # index each block's atoms by name; a name repeated within one block is ambiguous
        # (cannot be a 1:1 core atom), so it is never treated as core.
        per_block_by_name: dict[str, dict[str, list]] = {}
        for bname, st in blocks_struct.items():
            by_name: dict[str, list] = {}
            for a in st.atoms:
                by_name.setdefault(a.name, []).append(a)
            per_block_by_name[bname] = by_name

        def _unique_in_all(nm: str) -> bool:
            return all(len(per_block_by_name[b].get(nm, [])) == 1 for b in blocks_struct)

        # CORE names, in reference order: present exactly once in EVERY block, ONE shared LJ.
        # We compare Lennard-Jones parameters (element + sigma + epsilon), NOT the atom-TYPE
        # string: after an OpenMM->ParmEd hop, type names are SYNTHESIZED per structure, so a
        # physically identical backbone atom (e.g. the carbonyl O) gets different type strings in
        # separately-ingested variants -- which would wrongly kick it out of the shared core.
        # (Env atoms are unscaled, so they must carry one LJ; LJ equality is the real criterion.)
        core_names: list[str] = []
        seen: set[str] = set()
        for a in reference.atoms:
            nm = a.name
            if nm in seen:
                continue
            seen.add(nm)
            if not _unique_in_all(nm):
                continue                                    # missing / duplicated somewhere
            lj = {_lj_key(per_block_by_name[b][nm][0]) for b in blocks_struct}
            if len(lj) == 1:                                # one shared LJ => may be shared env
                core_names.append(nm)
        core_name_set = set(core_names)

        # hybrid index layout: env (core, reference order) first, then blocks appended
        env_atoms: set[int] = set()
        atom_names: dict[int, str] = {}
        source_names: dict[int, str] = {}
        name_to_hybrid: dict[str, int] = {}
        hybrid = 0
        for nm in core_names:
            name_to_hybrid[nm] = hybrid
            env_atoms.add(hybrid)
            atom_names[hybrid] = nm
            source_names[hybrid] = nm
            hybrid += 1

        used_names = set(atom_names.values())
        blocks: dict[str, set[int]] = {}
        core_corresp: dict[str, dict[int, int]] = {}
        variant_index_maps: dict[str, dict[int, int]] = {}
        for bname, st in blocks_struct.items():
            idx_map: dict[int, int] = {}
            corr: dict[int, int] = {}
            block_atoms: set[int] = set()
            # key maps by POSITION within the structure (0-based), not atom.idx: a mapper is
            # run on standalone variant structures (idx = 0..N-1) AND on a view over an
            # already-embedded reference residue (idx = its global indices). Position is the
            # only index both agree on; the caller resolves position -> its own atom.
            for pos, a in enumerate(st.atoms):
                is_core = (a.name in core_name_set
                           and len(per_block_by_name[bname][a.name]) == 1)
                if is_core:
                    h = name_to_hybrid[a.name]
                    corr[pos] = h                            # variant core atom -> env index
                else:
                    h = hybrid
                    hybrid += 1
                    block_atoms.add(h)
                    source_names[h] = a.name
                    hn = self._unique_name(a.name, used_names) if self.rename_blocks else a.name
                    used_names.add(hn)
                    atom_names[h] = hn
                idx_map[pos] = h
            blocks[bname] = block_atoms
            core_corresp[bname] = corr
            variant_index_maps[bname] = idx_map

        mutated = [nm for nm in self.backbone_atoms if nm not in core_name_set]
        metadata = {
            "reference": self.reference_name,
            "core_names": core_names,
            "backbone_mutation": bool(mutated),
            "mutated_backbone_atoms": mutated,
        }

        return CoreMapping(
            env_atoms=env_atoms, blocks=blocks,
            core_corresp=core_corresp, atom_names=atom_names,
            source_names=source_names, variant_index_maps=variant_index_maps,
            metadata=metadata,
        ).validate()

    @staticmethod
    def _unique_name(base: str, used: set[str]) -> str:
        """First name of the form base, base1, base2, ... not already used."""
        if base not in used:
            return base
        i = 1
        while f"{base}{i}" in used:
            i += 1
        return f"{base}{i}"
