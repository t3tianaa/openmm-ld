"""HybridSystemAssembler.

One object that runs the assembly substeps in order and returns a bundle:

    2.1 HybridStructureBuilder    graft variant blocks onto the env  (structure + mapping)
    2.2 ChargeRenormalization     average env + renormalize blocks to integer net (in place)
    2.3 solvate (optional)        water-box overlay + neutralizing ions   
    2.4 HybridSystemBuilder       hybrid Structure -> plain OpenMM System

    OUTPUT: HybridSystemBundle { structure, system, mapping }  (+ optional save)

The hub owns the swappable pieces (mapper / aligner / renormalizer / system_builder) so the
whole pipeline is configured in one place. It ends exactly where the engine begins: the bundle
carries the two objects the engine uses -- a plain System and (via the blueprint) an MSLDModel.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import openmm as mm
from openmm import app, unit

from ..mapping.core import HybridBlueprint
from .structure import HybridStructureBuilder
from .charges import SpreadRenormalizer
from .system import HybridSystemBuilder, is_solvated, strip_solvent


@dataclass
class HybridSystemBundle:
    """The product of assembly: a plain System + its hybrid Structure + the blueprint."""
    structure: object                       # pmd.Structure (solute [+ solvent], renormalized)
    system: object                          # plain OpenMM System (not yet lambda-scaled)
    mapping: HybridBlueprint                # combined-index CoreMapping per site
    reports: dict = field(default_factory=dict)   # per-site RenormReport, solvation info, ...

    def to_model(self, **kw):
        """The finalized MSLDModel for the engine (model_from_mappings under the hood)."""
        return self.mapping.to_model(**kw)

    def save(self, out_dir, *, prefix="hybrid", hyb_resname="HYB") -> dict:
        """Write the hybrid system's artifacts. Returns the paths dict.

          {prefix}_system.xml    OpenMM System (the plain, not-yet-lambda-scaled System)
          {prefix}_mapping.json  the HybridBlueprint (which atoms are env / each block, per site)
          {prefix}_topology.pdb  the topology with each mutated site MERGED into ONE residue
                                 (`hyb_resname`), on the site's chain, with CONECT (a novel residue)
          {prefix}_residue.xml   a self-contained OpenMM forcefield XML for the hybrid residue(s)
          {prefix}_start.pdb     the topology in the System's atom order (`openmm_ld.load_prepared`)
          {prefix}_positions.npy positions in nm, in full precision
        """
        from .hybrid_output import write_hybrid_pdb, write_hybrid_ff_xml
        os.makedirs(out_dir, exist_ok=True)
        paths = {
            "system": os.path.join(out_dir, f"{prefix}_system.xml"),
            "mapping": os.path.join(out_dir, f"{prefix}_mapping.json"),
            "topology": os.path.join(out_dir, f"{prefix}_topology.pdb"),
            "residue_ff": os.path.join(out_dir, f"{prefix}_residue.xml"),
            "start": os.path.join(out_dir, f"{prefix}_start.pdb"),
            "positions": os.path.join(out_dir, f"{prefix}_positions.npy"),
        }
        with open(paths["system"], "w") as f:
            f.write(mm.XmlSerializer.serialize(self.system))
        with open(paths["mapping"], "w") as f:
            f.write(self.mapping.to_json(indent=2))
        write_hybrid_pdb(self.structure, self.mapping, paths["topology"], hyb_resname=hyb_resname)
        write_hybrid_ff_xml(self.structure, self.mapping, paths["residue_ff"], hyb_resname=hyb_resname)
        positions = self.structure.positions
        with open(paths["start"], "w") as f:
            app.PDBFile.writeFile(self.structure.topology, positions, f)
        np.save(paths["positions"], np.array(positions.value_in_unit(unit.nanometer)))
        return paths


def endpoint_net_charge(structure, site_mappings) -> float:
    """Net charge of the hybrid's REFERENCE endpoint: every non-block atom + the first block
    (the reference variant) of each site. Summing all atoms instead counts each site's charge
    once per variant, which picks the wrong number of counter-ions whenever the blocks carry
    net charge. Warns if a site's variants differ in net charge (endpoints not iso-charge)."""
    import warnings
    q = [a.charge for a in structure.atoms]
    in_block = set()
    total = 0.0
    for m in site_mappings:
        nets = {name: sum(q[i] for i in atoms) for name, atoms in m.blocks.items()}
        for atoms in m.blocks.values():
            in_block.update(atoms)
        if nets:
            total += next(iter(nets.values()))
            if max(nets.values()) - min(nets.values()) > 0.5:
                warnings.warn(
                    "site variants differ in net charge (" +
                    ", ".join(f"{k}={v:+.2f}" for k, v in nets.items()) +
                    "); neutralizing the reference endpoint only", stacklevel=2)
    return total + sum(qi for i, qi in enumerate(q) if i not in in_block)


class HybridSystemAssembler:
    """Run structure -> charges -> (solvate) -> system, and return a HybridSystemBundle."""

    def __init__(self, *, mapper=None, aligner=None, renormalizer=None, system_builder=None):
        self.structure_builder = HybridStructureBuilder(mapper=mapper, aligner=aligner)
        self.renormalizer = renormalizer if renormalizer is not None else SpreadRenormalizer()
        self.system_builder = system_builder or HybridSystemBuilder()

    def assemble(self, env_structure, sites, *, renormalize=True,
                 solvate=False, solvate_kwargs=None,
                 restrip_solvent=True, fix_clashes=True) -> HybridSystemBundle:
        """Assemble the hybrid system.

        renormalize : run ChargeRenormalization (env averaging + block renorm) per site.
        solvate     : overlay a water box + neutralizing ions after charge renorm (§16.5).
        solvate_kwargs : passed to prep.solvate.solvate (padding, water_ff, ion_conc, ...).
        restrip_solvent : if the input env already carries water/ions, strip it before
                          assembly (assembly runs on a DRY solute).
        fix_clashes : after grafting, try the staggered rotamers of every grafted block that
                      clashes with the environment (clash.relieve_clashes) and keep the best one.
                      Every remaining clash is in reports["clashes"]; one deeper than
                      clash.WARN_OVERLAP is also warned about. LambdaDynamics.minimize() relaxes
                      and checks the endpoints later.
        """
        reports: dict = {}

        # a pre-solvated env is stripped so the graft happens on a dry solute (§16.5)
        if restrip_solvent and is_solvated(env_structure):
            env_structure = strip_solvent(env_structure)
            reports["stripped_input_solvent"] = True

        # 2.1 structure
        hres = self.structure_builder.build(env_structure, sites)
        structure = hres.structure

        # 2.1b clashes of the grafted blocks with the environment (before solvation: water is
        # placed around whatever geometry the blocks end up in)
        from .clash import block_clashes, relieve_clashes, format_clash
        clashes = (relieve_clashes(structure, hres.site_mappings) if fix_clashes
                   else block_clashes(structure, hres.site_mappings))
        reports["clashes"] = clashes
        left = [r for r in clashes if r.severe]
        if left:
            import warnings
            warnings.warn("grafted blocks clash with the environment (minimize() relaxes each "
                          "endpoint and checks it): " + "; ".join(format_clash(structure, r)
                                                                   for r in left), stacklevel=2)

        # 2.2 charges (per site; each site's own variants drive the env averaging)
        if renormalize:
            reports["renorm"] = {}
            for spec, mapping in zip(sites, hres.site_mappings):
                reports["renorm"][spec.reference_name] = self.renormalizer.apply(
                    structure, mapping, variants=spec.variants)

        # 2.3 solvate (optional). Solute atoms keep their indices (solute is added first), so
        # the site mappings stay valid; only the total atom count grows.
        if solvate:
            from .solvate import solvate as _solvate
            kw = dict(solvate_kwargs or {})
            kw.setdefault("net_charge", endpoint_net_charge(structure, hres.site_mappings))
            sres = _solvate(structure, **kw)
            structure = sres.structure
            reports["solvate"] = sres

        blueprint = HybridBlueprint(n_atoms=len(structure.atoms),
                                    sites=hres.site_mappings).validate()

        # 2.4 system
        system = self.system_builder.build(structure, blueprint)
        return HybridSystemBundle(structure=structure, system=system,
                                  mapping=blueprint, reports=reports)
