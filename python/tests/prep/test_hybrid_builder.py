"""Phase D: HybridStructureBuilder — the GATE, on a real system (MDM2 peptide, Amber ff14SB).

The load-bearing correctness test of the whole builder:

    take a real parametrized peptide, treat one residue (ALA4) as a mutation site with a real
    different side chain (SER, donated from another residue and aligned onto the backbone),
    assemble the hybrid, apply MSLD scaling, and check that with the WILD-TYPE fully on
    (lambda_WT=1, lambda_SER=0) the hybrid reproduces the untouched peptide's energy exactly.

That proves the builder (a) carved the WT side chain out of the environment correctly, (b)
left the shared core (backbone + CB) untouched, (c) scaled the WT block back to full strength
faithfully, and (d) turned the SER block fully off. The mapper (name+type core), the aligner
(Kabsch), and the term transplant are all exercised on real Amber data.

Uses examples/mdm2_peptide.pdb (standard residues). Variants are built here from the
peptide's own residues (no external mutation tool): the SER side chain is donated from SER2 and
the BackboneAligner superposes it onto ALA4's backbone.
"""
import copy
import os
import sys
import warnings

import numpy as np
import openmm as mm
import parmed as pmd
from openmm import app, unit

from openmm_ld.prep import HybridStructureBuilder, SiteSpec, AminoAcidBackboneMapper
from openmm_ld.builder import scale_msld_system, set_lambdas

_REF = mm.Platform.getPlatform("Reference")
_KJ = unit.kilojoule_per_mole
_PDB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "mdm2", "mdm2_peptide.pdb")
_cache = {}


def _fixture():
    """Parametrized, clash-minimized peptide + the ALA4 site + a SER donor side chain."""
    if "st" not in _cache:
        pdb = app.PDBFile(_PDB)
        ff = app.ForceField("amber14-all.xml")
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.addHydrogens(ff)                              # repair the bare N-terminus
        sysm = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff)
        ctx = mm.Context(sysm, mm.VerletIntegrator(0.001), _REF)
        ctx.setPositions(mod.positions)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=200)   # clear PDB clashes
        xyz = np.array(ctx.getState(getPositions=True).getPositions()
                       .value_in_unit(unit.nanometer)) * 10.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = pmd.openmm.load_topology(mod.topology, system=sysm, xyz=xyz)
        ala4 = [a.idx for a in list(st.residues)[3].atoms]          # ALA (4th residue)
        ser_src = [a.idx for a in list(st.residues)[1].atoms]       # SER donor (2nd residue)
        _cache.update(st=st, ALA4=ala4, SER=_extract(st, ser_src))
    return _cache["st"], _cache["ALA4"], _cache["SER"]


def _extract(st, indices):
    """A standalone Structure holding only `indices` (+ the terms among them)."""
    sub = copy.copy(st)
    keep = set(indices)
    mask = np.array([i not in keep for i in range(len(st.atoms))])
    sub.strip(mask)
    return sub


def _energy(system, coords_ang, lambdas=None):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), _REF)
    ctx.setPositions((np.array(coords_ang) / 10.0).tolist() * unit.nanometer)
    if lambdas is not None:
        set_lambdas(ctx, lambdas)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(_KJ)


def _build():
    st, ala4, ser = _fixture()
    res = HybridStructureBuilder(mapper=AminoAcidBackboneMapper(reference_name="WT")).build(
        st, [SiteSpec(residue_atoms=ala4, variants={"SER": ser}, reference_name="WT")])
    return st, ala4, res


# --------------------------------------------------------------- the GATE
def test_wt_on_reproduces_wt_peptide_exactly():
    st, ala4, res = _build()
    e_wt = _energy(st.createSystem(nonbondedMethod=app.NoCutoff), st.coordinates)

    model = res.blueprint.to_model(add_restraints=False)      # CATS off: exact energy compare
    system = res.structure.createSystem(nonbondedMethod=app.NoCutoff)
    scale_msld_system(system, model)
    coords = res.structure.coordinates

    e_wt_on = _energy(system, coords, [1.0, 1.0, 0.0])        # WT on, SER off
    rel = abs(e_wt_on - e_wt) / max(1.0, abs(e_wt))
    assert rel < 1e-9, (e_wt_on, e_wt, rel)

    # SER on is a genuinely different, finite end state (the block really changed the physics)
    e_ser_on = _energy(system, coords, [1.0, 0.0, 1.0])
    assert np.isfinite(e_ser_on) and abs(e_ser_on - e_wt_on) > 1.0
    print(f"  WT-on={e_wt_on:.5f}  WT peptide={e_wt:.5f}  rel={rel:.1e}  SER-on={e_ser_on:.1f}")


# --------------------------------------------------------------- assembly correctness
def test_structure_and_mapping():
    st, ala4, res = _build()
    cm = res.site_mappings[0]

    # shared core = backbone + CB (both sp3 C); WT block = ALA's methyl H's; SER's own atoms
    env_names = {cm.source_names[i] for i in cm.env_atoms}
    assert {"N", "CA", "C", "O", "CB"} <= env_names
    assert sorted(cm.source_names[i] for i in cm.blocks["WT"]) == ["HB1", "HB2", "HB3"]
    ser_block = {cm.source_names[i] for i in cm.blocks["SER"]}
    assert {"OG", "HG"} <= ser_block                         # SER's unique atoms are in its block
    assert cm.metadata["backbone_mutation"] is False

    # WT block atoms are the EXISTING env atoms; SER block atoms were APPENDED
    assert cm.blocks["WT"] <= set(ala4)
    assert min(cm.blocks["SER"]) >= len(st.atoms)            # appended past the env
    assert len(res.structure.atoms) == len(st.atoms) + len(cm.blocks["SER"])

    # the blueprint builds a 2-block, single-site model
    model = res.blueprint.to_model(add_restraints=False)
    assert model.n_blocks == 3 and model.n_sites == 2 and model.blocks_per_site == [1, 2]


def test_alignment_places_block_near_the_site():
    st, ala4, res = _build()
    cm = res.site_mappings[0]
    ca = next(st.atoms[i] for i in ala4 if st.atoms[i].name == "CA")
    ser_atoms = [res.structure.atoms[i] for i in cm.blocks["SER"]]
    nearest = min(((a.xx - ca.xx) ** 2 + (a.xy - ca.xy) ** 2 + (a.xz - ca.xz) ** 2) ** 0.5
                  for a in ser_atoms)
    # the SER side chain was superposed onto ALA4's backbone (a few A away), NOT left at the
    # donor residue's position (which is many A away across the peptide)
    assert nearest < 5.0, nearest


def test_same_site_blocks_are_excluded():
    st, ala4, res = _build()
    cm = res.site_mappings[0]
    model = res.blueprint.to_model(add_restraints=False)
    system = res.structure.createSystem(nonbondedMethod=app.NoCutoff)
    scale_msld_system(system, model)

    # MSLDNonbondedForce excludes different blocks of one site by construction, from their blocks and sites
    import msldplugin
    from openmm_ld.forces import PLUGIN_FORCE_NAME
    force = next(f for f in system.getForces() if f.getName() == PLUGIN_FORCE_NAME)
    force = msldplugin.MSLDNonbondedForce.cast(force)
    blocks = {i: force.getParticleParameters(i)[3] for i in cm.blocks["WT"] | cm.blocks["SER"]}
    for a in cm.blocks["WT"]:
        for c in cm.blocks["SER"]:
            assert model.classify_pair(a, c) == ("exclude",), (a, c)
            assert blocks[a] != blocks[c] and blocks[a] != 0 and blocks[c] != 0
            assert force.getBlockSite(blocks[a]) == force.getBlockSite(blocks[c])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nPhase D hybrid-builder tests passed")
