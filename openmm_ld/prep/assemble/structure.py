"""HybridStructureBuilder — graft variant blocks onto the shared environment.

This is the assembler: it turns (env structure + per-site variant residues + a mapping) into
one hybrid ParmEd Structure whose atoms/terms the engine can then scale. It ties together the
Phase B mapper (which atoms are env vs block) and the Phase C aligner (place each variant's
block in the reference frame), and reuses the tested term-transplant from `merge.py`.

Per site, for the residue being mutated:

    env structure ─┐   the site residue's BACKBONE (+ type-matched atoms) is env; its own
                   │   side chain becomes the reference/WT block (already in the structure).
    variant resid ─┘   each OTHER variant is aligned onto that backbone, its block atoms are
                       APPENDED (aligned coords, renamed), and only its block-touching bonded
                       terms are transplanted (remapped: core atom -> env index, block -> new).

    ●─●─●  backbone (env, stays)          the WT side chain is marked as a block (not moved);
        │                                 each other variant's side chain rides its backbone
    ○   WT side chain (block, in place)   fit onto the env backbone, then is appended.
        ◦   variant side chain (block, appended after alignment)

Output: a hybrid `Structure` + a `HybridBlueprint` (one combined-index `CoreMapping` per site)
that feeds `model_from_mappings` / `MSLDModel`. Charges are left untouched here — env-charge
averaging and block renormalization are the ChargeRenormalization step (Phase E).

Frames: the Phase B mapper emits a residue-local CoreMapping; this builder offsets it into the
full system, so the returned CoreMapping's indices are global (into the hybrid Structure).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..mapping.core import CoreMapping, HybridBlueprint
from .align import BackboneAligner
from .merge import _copy_atom, _transplant_terms, _TypeCache


class _View:
    """Minimal structure-like wrapper over a list of atoms (mapper/aligner read .atoms)."""
    def __init__(self, atoms):
        self.atoms = atoms


@dataclass
class SiteSpec:
    """One mutation site.

    residue_atoms  : global indices in the env structure of the site (WT) residue, in order.
    variants       : {name: variant residue Structure} for the NON-WT candidates. Each holds
                     the residue backbone (for fit + correspondence) + its side chain.
    reference_name : block label for the wild-type side chain (already in the env structure).
    mapper         : REQUIRED, here or builder-wide (no default, explicit by design):
                     AminoAcidBackboneMapper (peptides) / MCSMapper (ligands) / ManualMapper.
    aligner        : optional override (default: BackboneAligner).
    """
    residue_atoms: list[int]
    variants: dict[str, object]
    reference_name: str = "WT"
    mapper: object = None
    aligner: object = None


@dataclass
class HybridResult:
    structure: object                       # pmd.Structure (env with blocks grafted)
    blueprint: HybridBlueprint              # one combined-index CoreMapping per site
    site_mappings: list = field(default_factory=list)


class HybridStructureBuilder:
    """Assemble a hybrid Structure from an env structure and per-site variant residues."""

    def __init__(self, mapper=None, aligner=None):
        self._mapper = mapper
        self._aligner = aligner

    def build(self, env_structure, sites) -> HybridResult:
        hybrid = copy.copy(env_structure)       # ParmEd deep-copies atoms/terms/types
        site_mappings = [self._build_site(hybrid, spec) for spec in sites]
        blueprint = HybridBlueprint(n_atoms=len(hybrid.atoms), sites=site_mappings).validate()
        return HybridResult(structure=hybrid, blueprint=blueprint, site_mappings=site_mappings)

    # ---- one site --------------------------------------------------
    def _build_site(self, hybrid, spec: SiteSpec) -> CoreMapping:
        ref_name = spec.reference_name
        site_atoms = [hybrid.atoms[i] for i in spec.residue_atoms]   # WT residue atoms (global)

        # the mapper is ALWAYS explicit (no peptide-vs-ligand guessing): the caller states which
        # mapper each site uses, via SiteSpec(mapper=...) or a builder-wide default.
        mapper = spec.mapper or self._mapper
        if mapper is None:
            raise ValueError(
                "no mapper for this site: pass one explicitly via SiteSpec(mapper=...) or "
                "HybridStructureBuilder/HybridSystemAssembler(mapper=...). Use "
                "AminoAcidBackboneMapper for peptide/residue mutations, MCSMapper for ligands.")
        aligner = spec.aligner or self._aligner or BackboneAligner()

        rlmap = mapper.get_mapping(reference=_View(site_atoms), variants=spec.variants)
        vim = rlmap.variant_index_maps                     # {name: {var_local -> hybrid_local}}

        # --- reference (WT): its side chain is a block that is ALREADY in the structure ---
        wt_block_hyb = rlmap.blocks[ref_name]
        wt_block_globals: set[int] = set()
        env_core_globals: set[int] = set()
        hyb_core_to_global: dict[int, int] = {}            # residue-local core idx -> env global
        ref_local_to_global: dict[int, int] = {}
        for ref_local, hyb in vim[ref_name].items():
            g = site_atoms[ref_local].idx
            ref_local_to_global[ref_local] = g
            if hyb in wt_block_hyb:
                wt_block_globals.add(g)
            else:
                env_core_globals.add(g)
                hyb_core_to_global[hyb] = g

        blocks_global: dict[str, set[int]] = {ref_name: wt_block_globals}
        atom_names_g: dict[int, str] = {a.idx: a.name for a in site_atoms}
        source_names_g: dict[int, str] = {a.idx: a.name for a in site_atoms}
        combined_vim: dict[str, dict[int, int]] = {ref_name: dict(ref_local_to_global)}

        # --- every other variant: align, append its block, transplant its block terms ---
        for name, vstruct in spec.variants.items():
            T = aligner.align(_View(site_atoms), vstruct)
            block_hyb = rlmap.blocks[name]
            resnum = len(hybrid.residues) + 1               # all block atoms in ONE new residue
            appended: set[int] = set()
            block_src_local: set[int] = set()
            v_local_to_g: dict[int, int] = {}
            for v_local, hyb in vim[name].items():
                if hyb in block_hyb:                        # block atom -> append (aligned)
                    src = vstruct.atoms[v_local]
                    new_atom = _copy_atom(src)
                    x, y, z = T.apply([src.xx, src.xy, src.xz])
                    new_atom.xx, new_atom.xy, new_atom.xz = float(x), float(y), float(z)
                    new_atom.name = rlmap.atom_names.get(hyb, src.name)
                    # keep appended block atoms on the SITE residue's chain (not chain-less), so the
                    # raw Structure isn't misleading; hybrid_output.py merges them into one residue.
                    hybrid.add_atom(new_atom, name, resnum, chain=site_atoms[0].residue.chain)
                    g = new_atom.idx
                    appended.add(g)
                    block_src_local.add(v_local)
                    atom_names_g[g] = new_atom.name
                    source_names_g[g] = rlmap.source_names.get(hyb, src.name)
                else:                                       # core atom -> shares the env index
                    g = hyb_core_to_global[hyb]
                v_local_to_g[v_local] = g
            blocks_global[name] = appended
            combined_vim[name] = v_local_to_g
            _transplant_terms(hybrid, vstruct, v_local_to_g, block_src_local, _TypeCache())

        return CoreMapping(
            env_atoms=env_core_globals, blocks=blocks_global,
            atom_names=atom_names_g, source_names=source_names_g,
            variant_index_maps=combined_vim, metadata=dict(rlmap.metadata),
        ).validate()
