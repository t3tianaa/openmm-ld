"""Phase F + G: HybridSystemBuilder + HybridSystemAssembler (the hub) + bundle/save.

End-to-end on the real MDM2 peptide (Amber ff14SB):
  - the hub runs structure -> charges -> (solvate) -> system and returns a bundle,
  - with the WILD-TYPE block fully on the assembled (dry) System reproduces the untouched
    peptide energy exactly -- the Phase D gate, now through the whole hub,
  - a solvated bundle yields a finite PME energy and a neutral, boxed system,
  - the bundle saves to system.xml / mapping.json / topology.pdb and round-trips,
  - the same-site cross-variant bonded-term guard fires, and solvent detect/strip work.
"""
import copy
import os
import sys
import tempfile
import warnings

import numpy as np
import openmm as mm
import parmed as pmd
from openmm import app, unit

from openmm_ld.prep import (HybridSystemAssembler, HybridSystemBuilder, HybridBlueprint,
                       CoreMapping, SiteSpec, AminoAcidBackboneMapper,
                       is_solvated, strip_solvent)
from openmm_ld.builder import scale_msld_system, set_lambdas

_REF = mm.Platform.getPlatform("Reference")
_KJ = unit.kilojoule_per_mole
_PDB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "mdm2", "mdm2_peptide.pdb")
_cache = {}


def _fixture():
    if "st" not in _cache:
        pdb = app.PDBFile(_PDB)
        ff = app.ForceField("amber14-all.xml")
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.addHydrogens(ff)
        sysm = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff)
        ctx = mm.Context(sysm, mm.VerletIntegrator(0.001), _REF)
        ctx.setPositions(mod.positions)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=200)
        xyz = np.array(ctx.getState(getPositions=True).getPositions()
                       .value_in_unit(unit.nanometer)) * 10.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = pmd.openmm.load_topology(mod.topology, system=sysm, xyz=xyz)
        ala4 = [a.idx for a in list(st.residues)[3].atoms]
        ser = _extract(st, [a.idx for a in list(st.residues)[1].atoms])
        e_wt = _energy(st.createSystem(nonbondedMethod=app.NoCutoff), st.coordinates)
        _cache.update(st=st, spec=SiteSpec(residue_atoms=ala4, variants={"SER": ser},
                                           reference_name="WT"), e_wt=e_wt)
    return _cache["st"], _cache["spec"], _cache["e_wt"]


def _extract(st, idx):
    sub = copy.copy(st)
    keep = set(idx)
    sub.strip(np.array([i not in keep for i in range(len(st.atoms))]))
    return sub


def _energy(system, coords_ang, lambdas=None):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), _REF)
    ctx.setPositions((np.array(coords_ang) / 10.0).tolist() * unit.nanometer)
    if lambdas is not None:
        set_lambdas(ctx, lambdas)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(_KJ)


# --------------------------------------------------------------- the gate, through the hub
def test_bundle_wt_on_reproduces_peptide():
    st, spec, e_wt = _fixture()
    bundle = HybridSystemAssembler(mapper=AminoAcidBackboneMapper(reference_name="WT")).assemble(
        st, [spec], renormalize=False, solvate=False)
    assert bundle.structure.box is None                      # dry -> auto NoCutoff

    model = bundle.to_model(add_restraints=False)
    scale_msld_system(bundle.system, model)
    e_on = _energy(bundle.system, bundle.structure.coordinates, [1.0, 1.0, 0.0])
    assert abs(e_on - e_wt) / max(1.0, abs(e_wt)) < 1e-9, (e_on, e_wt)


# --------------------------------------------------------------- solvation -> finite PME
def test_solvated_bundle_pme_is_finite():
    st, spec, _ = _fixture()
    bundle = HybridSystemAssembler(mapper=AminoAcidBackboneMapper(reference_name="WT")).assemble(
        st, [spec], renormalize=True, solvate=True, solvate_kwargs=dict(padding=0.7))
    assert bundle.structure.box is not None                  # solvated -> periodic
    assert abs(bundle.reports["solvate"].net_charge) < 0.5   # neutralized
    assert bundle.reports["renorm"]["WT"].env is not None     # env averaging happened (ALA vs SER)

    model = bundle.to_model(add_restraints=False)
    scale_msld_system(bundle.system, model)                  # PME path (box present)
    e = _energy(bundle.system, bundle.structure.coordinates, [1.0, 1.0, 0.0])
    assert np.isfinite(e)


# --------------------------------------------------------------- save round-trip
def test_bundle_save_round_trip():
    st, spec, _ = _fixture()
    bundle = HybridSystemAssembler(mapper=AminoAcidBackboneMapper(reference_name="WT")).assemble(
        st, [spec], renormalize=False, solvate=False)
    with tempfile.TemporaryDirectory() as d:
        paths = bundle.save(d)
        sys_back = mm.XmlSerializer.deserialize(open(paths["system"]).read())
        assert sys_back.getNumParticles() == bundle.system.getNumParticles()
        bp_back = HybridBlueprint.from_json(open(paths["mapping"]).read())
        assert len(bp_back.sites) == 1
        assert bp_back.sites[0].blocks.keys() == bundle.mapping.sites[0].blocks.keys()
        assert os.path.exists(paths["topology"])


# --------------------------------------------------------------- guards & solvent helpers
def test_cross_variant_bond_is_refused():
    # a bond between two blocks of the SAME site is illegal; the system builder must refuse it
    st = pmd.Structure()
    st.add_atom(pmd.Atom(name="A"), "R", 1)
    st.add_atom(pmd.Atom(name="B"), "R", 1)
    st.bonds.append(pmd.Bond(st.atoms[0], st.atoms[1]))
    bp = HybridBlueprint(n_atoms=2, sites=[CoreMapping(env_atoms=set(),
                                                       blocks={"A": {0}, "B": {1}})])
    try:
        HybridSystemBuilder().build(st, bp)
        assert False, "expected a cross-variant guard error"
    except ValueError as ex:
        assert "cross-variant" in str(ex)


def test_solvent_detect_and_strip():
    st = pmd.Structure()
    st.add_atom(pmd.Atom(name="C"), "SOL_UTE", 1)               # a solute atom
    for nm, z in [("O", 8), ("H1", 1), ("H2", 1)]:             # a water residue
        st.add_atom(pmd.Atom(name=nm, atomic_number=z), "HOH", 2)
    ion = pmd.Atom(name="NA", atomic_number=11)
    ion.charge = 1.0
    st.add_atom(ion, "NA", 3)                                  # an ion residue
    assert is_solvated(st)
    dry = strip_solvent(st)
    assert not is_solvated(dry)
    assert len(dry.atoms) == 1 and dry.atoms[0].name == "C"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nPhase F+G hybrid-system tests passed")
