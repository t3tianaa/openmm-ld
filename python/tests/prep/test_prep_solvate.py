"""P4: solvation + ions + box (water-box overlay).

We solvate a small dry solute and check the essentials: a periodic box is set, water was
added, the system is NEUTRAL (counter-ions added for a charged solute), no water clashes
with the solute, and createSystem(PME) gives a finite energy.
"""
import os
import sys

import numpy as np
import openmm as mm
import parmed as pmd
from openmm import app, unit

from openmm_ld.prep import solvate

_REF = mm.Platform.getPlatform("Reference")
_NM = unit.nanometer


def _solute(net_charge=0.0):
    """A small 3-atom solute; the last atom carries `net_charge`."""
    s = pmd.Structure()
    charges = [0.0, 0.0, net_charge]
    for i in range(3):
        a = pmd.Atom(name=f"A{i}", atomic_number=6, charge=charges[i], mass=12.0, type=f"CX{i}")
        at = pmd.AtomType(f"CX{i}", i + 1, 12.0, 6); at.set_lj_params(0.1, 1.7)
        a.atom_type = at
        s.add_atom(a, "LIG", 1)
    bt = pmd.BondType(300.0, 1.5); s.bond_types.append(bt)
    s.bonds.append(pmd.Bond(s.atoms[0], s.atoms[1], type=bt))
    s.bonds.append(pmd.Bond(s.atoms[1], s.atoms[2], type=bt))
    s.coordinates = np.array([[10, 10, 10], [11.5, 10, 10], [13, 10, 10]], dtype=float)
    return s


def _pme_energy(structure, cutoff_nm=0.9):
    system = structure.createSystem(nonbondedMethod=app.PME,
                                    nonbondedCutoff=cutoff_nm * _NM)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), _REF)
    ctx.setPositions(structure.positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_solvate_neutral_solute():
    res = solvate(_solute(0.0), padding=0.8, ion_conc=0.0)
    assert res.n_waters > 0
    assert res.box_nm > 0 and res.structure.box is not None
    assert abs(res.net_charge) < 1e-6                       # already neutral, no ions
    assert res.n_pos_ions == 0 and res.n_neg_ions == 0
    assert np.isfinite(_pme_energy(res.structure))


def test_solvate_charged_solute_gets_neutralized():
    # +1 solute -> exactly one Cl- added, system neutral
    res = solvate(_solute(+1.0), padding=0.8, ion_conc=0.0)
    assert abs(res.net_charge) < 1e-6, res.net_charge
    assert res.n_neg_ions == 1 and res.n_pos_ions == 0
    assert np.isfinite(_pme_energy(res.structure))


def test_solvate_negative_solute_gets_sodium():
    res = solvate(_solute(-1.0), padding=0.8, ion_conc=0.0)
    assert abs(res.net_charge) < 1e-6
    assert res.n_pos_ions == 1 and res.n_neg_ions == 0


def test_no_water_clashes_with_solute():
    res = solvate(_solute(0.0), padding=0.8, clash=0.20)
    st = res.structure
    solute_idx = [a.idx for a in st.atoms if a.residue.name == "LIG"]
    water_idx = [a.idx for a in st.atoms if a.residue.name == "HOH"]
    su = np.array(st.coordinates)[solute_idx]
    wa = np.array(st.coordinates)[water_idx]
    mind = np.sqrt(((wa[:, None, :] - su[None, :, :]) ** 2).sum(-1).min(1)).min()
    assert mind >= 0.20 * 10.0 - 1e-6, mind / 10.0      # angstrom vs clash (nm)


def test_ionic_strength_adds_salt():
    res = solvate(_solute(0.0), padding=1.0, ion_conc=0.5)
    assert res.n_pos_ions >= 1 and res.n_neg_ions >= 1    # background salt present
    assert abs(res.net_charge) < 1e-6                     # still neutral


def test_charmm_water_lj_recovered():
    # CHARMM water LJ lives in a tabulated CustomNonbondedForce ParmEd drops; solvate must
    # recover it (exact for water) so water has real LJ and PME energy is finite.
    res = solvate(_solute(0.0), padding=0.8, water_ff=("charmm36/water.xml",),
                  water_model="tip3p")
    assert res.n_waters > 0 and np.isfinite(_pme_energy(res.structure))
    # water oxygen must have the CHARMM TIP3P epsilon (~0.152 kcal/mol = 0.636 kJ/mol),
    # NOT zero (which is what the old lossy conversion produced)
    o = next(a for a in res.structure.atoms
             if a.residue.name == "HOH" and a.atomic_number == 8)
    assert abs(o.epsilon - 0.1521) < 1e-3, o.epsilon          # kcal/mol (ParmEd units)


def test_charmm_ion_water_nbfix_preserved():
    # the whole CHARMM LJ energy (incl. ion-water NBFIX) must be reproduced by the ParmEd
    # recovery; if NBFIX were dropped to Lorentz-Berthelot, the ion-water pairs would differ.
    import warnings
    import parmed as pmd
    from openmm_ld.prep import recover_charmm_lj

    ff = app.ForceField("charmm36/water.xml")
    mod = app.Modeller(app.Topology(), np.zeros((0, 3)) * unit.nanometer)
    mod.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(2.2, 2.2, 2.2) * unit.nanometer,
                   ionicStrength=1.0 * unit.molar)          # plenty of ions
    xyz_ang = np.array(mod.positions.value_in_unit(unit.nanometer)) * 10.0
    sys0 = ff.createSystem(mod.topology, nonbondedMethod=app.PME, rigidWater=False)

    def charmm_lj_energy(system, positions):
        for f in system.getForces():
            f.setForceGroup(1 if isinstance(f, mm.CustomNonbondedForce) else 0)
        ctx = mm.Context(system, mm.VerletIntegrator(0.001), _REF)
        ctx.setPositions(positions)
        return ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    e_orig = charmm_lj_energy(sys0, mod.positions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = pmd.openmm.load_topology(mod.topology, system=sys0, xyz=xyz_ang)
    n_before = sum(len(a.atom_type.nbfix) for a in st.atoms if a.atom_type)
    recover_charmm_lj(st, sys0)
    n_after = sum(len(a.atom_type.nbfix) for a in st.atoms if a.atom_type)
    assert n_after > 0, "expected recovered ion-water NBFIX entries"

    sys1 = st.createSystem(nonbondedMethod=app.PME)
    assert st.has_NBFIX()                                    # NBFIX carried into ParmEd
    e_rec = charmm_lj_energy(sys1, st.positions)
    assert abs(e_rec - e_orig) < 1e-3 * max(1.0, abs(e_orig)), (e_rec, e_orig)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nP4 solvation tests passed")
