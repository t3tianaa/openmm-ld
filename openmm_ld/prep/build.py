"""build_msld_system — one-call façade: ingest every part + wire the sites + assemble.

The prep layer is deliberately per-part: `ingest` turns ONE file (or file set) into ONE
parametrized ParmEd Structure, and the assembler grafts the parts into a hybrid. That split is
FORCED by the force field -- the combined alchemical residue is not a real molecule and has no
template, so each real part is parametrized on its own and then grafted. This façade keeps the
split but removes the boilerplate: it ingests the environment + each variant (honouring a per-part
force field), resolves each site's residue to atom indices, picks the mapper, and calls the
assembler.

    build_msld_system(environment, sites, forcefield=[...]) -> HybridSystemBundle

Per-part force fields (why a Part carries its own list): a NON-STANDARD amino acid needs the
standard ffxml PLUS its own residue ffxml, and a ligand needs a small-molecule force field -- so a
single global force field is not enough. A Part can override the default entirely (`forcefield=`)
or extend it (`extra_forcefield=`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .ingest import ingest as _ingest
from .mapping import AminoAcidBackboneMapper, MCSMapper
from .assemble import HybridSystemAssembler, SiteSpec


@dataclass
class Part:
    """One molecule to ingest: a file path + how to parametrize it.

    forcefield       : the FULL force-field list for this part; None inherits build_msld_system's
                       default. A prmtop/psf is self-contained -- leave it None.
    extra_forcefield : ffxml(s) APPENDED to the effective list -- e.g. a non-standard residue's own
                       ffxml on top of the standard one: extra_forcefield=['phosphoser.xml'].
    add_hydrogens    : run Modeller.addHydrogens (e.g. a heavy-atom-only ENVIRONMENT PDB). Do NOT
                       use it on a lone monomer VARIANT: Modeller treats a single residue as a chain
                       terminus and adds terminal hydrogens (it matches e.g. NSER), which breaks the
                       clean in-chain monomer. Provide a variant residue WITH its in-chain hydrogens.
    ignore_external_bonds : match residue templates ignoring missing external bonds, so a single
                       amino-acid variant is provided as a plain MONOMER residue (no caps): a lone
                       in-chain residue's N/C lack their peptide-bond partners, so OpenMM would
                       otherwise raise "No template found ... missing a terminal capping group".
                       With this the mid-chain template matches, giving the correct IN-CHAIN
                       types/charges with no terminal OXT/H. Requires the residue's hydrogens to be
                       present already (see add_hydrogens). None (default) inherits the site default
                       -- True for a variant (it is a fragment), False for the environment.
    """
    path: str
    forcefield: list | None = None
    extra_forcefield: list | None = None
    add_hydrogens: bool = False
    ignore_external_bonds: bool | None = None


@dataclass
class Site:
    """One mutation site.

    residue        : residue INDEX in the environment (0-based, into env.residues), OR an explicit
                     list of atom indices in the environment.
    variants       : {name: path | Part | Structure} for the NON-WT candidates (a bare path inherits
                     the default force field). Any number of variants may compete at the site.
    mapper         : 'backbone' (peptide, by atom name+type) or 'mcs' (ligand, by graph MCS), or a
                     Mapper instance. Explicit by design -- no peptide-vs-ligand auto-detection.
    reference_name : label for the environment's own side chain at this site (it competes as a block).
    aligner        : optional aligner override (default: BackboneAligner).
    """
    residue: int | list
    variants: dict
    mapper: object = "backbone"
    reference_name: str = "WT"
    aligner: object = None


def _effective_ff(part: Part, default_ff):
    """Force-field list for a Part: (its own, or the default) + any extras. None if self-contained."""
    base = part.forcefield if part.forcefield is not None else default_ff
    extra = part.extra_forcefield or []
    if base is None and not extra:
        return None                                     # a prmtop/psf part needs no force field
    return list(base or []) + list(extra)


def _resolve_part(part, default_ff, default_addh, ignore_external_bonds=False):
    """Ingest a part. `part` is a path (str), a Part, or an already-parametrized Structure.

    `ignore_external_bonds` is the SITE-LEVEL default (variants pass True since each is a monomer
    fragment; the environment passes False). A Part overrides it via Part.ignore_external_bonds
    (None inherits this default). Ignored when `part` is already a Structure (nothing to ingest).
    """
    if isinstance(part, Part):
        ieb = (part.ignore_external_bonds if part.ignore_external_bonds is not None
               else ignore_external_bonds)
        return _ingest(part.path, forcefield=_effective_ff(part, default_ff),
                       add_hydrogens=part.add_hydrogens, ignore_external_bonds=ieb)
    if isinstance(part, (str, os.PathLike)):            # a path (str or pathlib.Path) -> ingest it
        return _ingest(part, forcefield=default_ff, add_hydrogens=default_addh,
                       ignore_external_bonds=ignore_external_bonds)
    return part                                         # already a parametrized Structure -> use as-is


def _resolve_mapper(mapper, reference_name):
    """A mapper shorthand ('backbone'/'mcs') -> a Mapper wired to reference_name; else pass through."""
    if isinstance(mapper, str):
        key = mapper.lower()
        if key == "backbone":
            return AminoAcidBackboneMapper(reference_name=reference_name)
        if key == "mcs":
            return MCSMapper(reference_name=reference_name)
        raise ValueError(f"unknown mapper '{mapper}'; use 'backbone', 'mcs', or a Mapper instance")
    if mapper is None:
        raise ValueError("each Site needs a mapper: 'backbone', 'mcs', or a Mapper instance")
    return mapper


def build_msld_system(environment, sites, *, forcefield=None, add_hydrogens: bool = False,
                      renormalize: bool = True, solvate: bool = False, solvate_kwargs=None,
                      fix_clashes: bool = True):
    """Ingest the environment + every variant, wire each site, and assemble the hybrid system.

    environment : path | Part | Structure -- the peptide / protein / environment.
    sites       : list of Site.
    forcefield  : default force field (list of ffxml names/paths) for parts that don't set their own.
                  Parts override (Part.forcefield) or extend it (Part.extra_forcefield), so a
                  non-standard residue carries its OWN ffxml on top of the standard one, and a ligand
                  variant carries a small-molecule force field.
    fix_clashes : try other rotamers for grafted blocks that clash with the environment
                  (see HybridSystemAssembler.assemble); bundle.reports["clashes"] lists them.
    Returns a HybridSystemBundle (bundle.to_model() -> MSLDModel; bundle.system -> plain System).
    """
    default_ff = list(forcefield) if forcefield else None
    # the environment is a full system (strict template matching); each variant is a MONOMER
    # fragment, so it is ingested with ignore_external_bonds=True (no caps needed -- a lone
    # residue's backbone N/C have no peptide-bond partners). A Part can override per part.
    env = _resolve_part(environment, default_ff, add_hydrogens, ignore_external_bonds=False)

    specs = []
    for site in sites:
        if isinstance(site.residue, int):
            residue_atoms = [a.idx for a in list(env.residues)[site.residue].atoms]
        else:
            residue_atoms = list(site.residue)
        variants = {name: _resolve_part(v, default_ff, add_hydrogens, ignore_external_bonds=True)
                    for name, v in site.variants.items()}
        specs.append(SiteSpec(
            residue_atoms=residue_atoms, variants=variants,
            reference_name=site.reference_name,
            mapper=_resolve_mapper(site.mapper, site.reference_name),
            aligner=site.aligner))

    return HybridSystemAssembler().assemble(
        env, specs, renormalize=renormalize, solvate=solvate, solvate_kwargs=solvate_kwargs,
        fix_clashes=fix_clashes)
