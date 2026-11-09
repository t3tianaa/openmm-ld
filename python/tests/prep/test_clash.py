"""Clash guards for grafted blocks (prep/assemble/clash.py) and the endpoint check
(LambdaDynamics.check_endpoints).

The structure builder places a variant's side chain in whatever rotamer its file has. On MDM2 the
grafted ILE put CD1 0.147 nm from MET38 SD: at partial lambda soft-core hides it, so minimization
leaves it there and lambda_ILE stays at 0. Verifies:
  - block_clashes finds a block atom sitting on an environment atom, and skips 1-2/1-3/1-4 pairs
    and polar-H ... acceptor contacts (hydrogen bonds are not clashes),
  - relieve_clashes rotates a clashing (non-reference) block to a staggered rotamer that clears
    the clash, keeps its bond lengths, and leaves the reference block in place,
  - check_endpoints warns / raises on an endpoint far above its site-mates and is silent otherwise.
"""
import os
import sys
import types
import warnings

import numpy as np
import openmm as mm
import parmed as pmd
from openmm import unit, Vec3

from openmm_ld.prep.assemble.clash import block_clashes, relieve_clashes
from openmm_ld.model import MSLDModel
from openmm_ld.builder import scale_msld_system
from openmm_ld.dynamics import LambdaDynamics

_REF = mm.Platform.getPlatform("Reference")
_NM = unit.nanometer


def _structure(atoms, bonds):
    """atoms: [(name, atomic_number, (x, y, z) in nm)]; bonds: [(i, j)]."""
    st = pmd.Structure()
    for name, z, xyz in atoms:
        a = pmd.Atom(name=name, atomic_number=z)
        a.xx, a.xy, a.xz = (10.0 * v for v in xyz)
        st.add_atom(a, "RES", 1)
    for i, j in bonds:
        st.bonds.append(pmd.Bond(st.atoms[i], st.atoms[j]))
    return st


def _site(structure):
    """env chain N(0)-P(1)-A(2); block WT = {3} on A, block VAR = {4} on A pointing at an env
    obstacle atom O(5) placed 0.10 nm away (a clash). Rotating VAR 120 deg about P->A clears it."""
    bl = 0.153
    P, A = np.array([0.0, 0.0, 0.0]), np.array([bl, 0.0, 0.0])
    # tetrahedral directions around A, perpendicular ring about the P->A axis
    t = np.radians(109.5)
    def sub(phi):
        return A + bl * np.array([-np.cos(t), np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi)])
    var = sub(0.0)
    away = (var - A) / np.linalg.norm(var - A)
    obstacle = var + 0.10 * away
    atoms = [("N", 7, (-0.15, 0.05, 0.0)), ("P", 6, tuple(P)), ("A", 6, tuple(A)),
             ("CW", 6, tuple(sub(np.radians(120)))), ("CV", 6, tuple(var)), ("O", 8, tuple(obstacle))]
    st = _structure(atoms, [(0, 1), (1, 2), (2, 3), (2, 4)])
    mapping = types.SimpleNamespace(blocks={"WT": {3}, "VAR": {4}})
    return st, mapping


def test_block_clashes_finds_the_overlap():
    st, m = _site(None)
    rep = {r.block: r for r in block_clashes(st, [m])}
    assert not rep["WT"].clashes
    w = rep["VAR"].worst
    assert w is not None and (w.atom, w.partner) == (4, 5)
    assert abs(w.distance - 0.10) < 1e-6 and w.contact > 0.2


def test_bonded_and_hbond_pairs_are_not_clashes():
    # O-H ... O hydrogen bond at 0.18 nm, and a 1-3 pair at 0.25 nm: neither is a clash
    atoms = [("C", 6, (0.0, 0.0, 0.0)), ("O1", 8, (0.143, 0.0, 0.0)), ("H1", 1, (0.143, 0.096, 0.0)),
             ("O2", 8, (0.143, 0.276, 0.0))]
    st = _structure(atoms, [(0, 1), (1, 2)])
    m = types.SimpleNamespace(blocks={"OH": {1, 2}})
    rep = block_clashes(st, [m])[0]
    assert not rep.clashes, [(c.atom, c.partner, c.distance) for c in rep.clashes]


def test_relieve_clashes_rotates_the_variant_only():
    st, m = _site(None)
    before = np.array([[a.xx, a.xy, a.xz] for a in st.atoms]) / 10.0
    reps = {r.block: r for r in relieve_clashes(st, [m])}
    after = np.array([[a.xx, a.xy, a.xz] for a in st.atoms]) / 10.0
    assert reps["VAR"].score_before > 0 and reps["VAR"].score_after == 0.0
    assert reps["VAR"].rotation                                  # a rotamer was applied
    assert np.allclose(after[[0, 1, 2, 3, 5]], before[[0, 1, 2, 3, 5]])   # env + reference fixed
    assert abs(np.linalg.norm(after[4] - after[2]) - 0.153) < 1e-9        # bond length kept
    assert np.linalg.norm(after[4] - after[5]) > 0.25                     # clash cleared
    assert not block_clashes(st, [m])[1].clashes


def _endpoint_system(clash: bool):
    """env atoms 0,1 ; site-1 blocks b1={2}, b2={3}. With `clash`, block-1 atom 2 sits 0.05 nm
    from env atom 0 (its endpoint far above b2's)."""
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.15, 200000.0); bond.addBond(1, 2, 0.15, 200000.0); bond.addBond(1, 3, 0.15, 200000.0)
    system.addForce(bond)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for _ in range(4):
        nb.addParticle(0.0, 0.30, 1.0)
    nb.addException(0, 1, 0.0, 0.1, 0.0); nb.addException(1, 2, 0.0, 0.1, 0.0); nb.addException(1, 3, 0.0, 0.1, 0.0)
    system.addForce(nb)
    x2 = Vec3(0.05, 0.0, 0.0) if clash else Vec3(0.15, -0.15, 0.0)
    pos = [Vec3(0, 0, 0), Vec3(0.15, 0, 0), x2, Vec3(0.15, 0.15, 0)] * _NM
    model = MSLDModel(4)
    model.add_block(site=1, atoms=[2]); model.add_block(site=1, atoms=[3]); model.finalize()
    scale_msld_system(system, model)
    return LambdaDynamics(system, model, pos, temperature=300.0, timestep=0.001, platform=_REF)


def test_check_endpoints_raises_and_warns_on_a_clash():
    dyn = _endpoint_system(clash=True)
    try:
        dyn.check_endpoints()
    except RuntimeError as ex:
        assert "block 1" in str(ex) and "atom 2-0" in str(ex), str(ex)
    else:
        raise AssertionError("expected RuntimeError for a clashing endpoint")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gaps = dyn.check_endpoints(warn_gap=1.0, max_gap=float("inf"))
    assert any("large endpoint gap" in str(x.message) for x in w)
    assert gaps[1] > 1e4 and gaps[2] == 0.0


def test_check_endpoints_silent_without_a_clash():
    dyn = _endpoint_system(clash=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gaps = dyn.check_endpoints()
    assert not [x for x in w if "endpoint" in str(x.message)]
    assert max(gaps.values()) < 1.0e3


def test_minimize_removes_the_clash_and_passes_the_check():
    dyn = _endpoint_system(clash=True)
    dyn.minimize(max_iterations=2000)                  # relax_blocks + check_endpoints by default
    assert max(dyn.endpoint_gaps.values()) < 1.0e3


if __name__ == "__main__":
    test_block_clashes_finds_the_overlap()
    test_bonded_and_hbond_pairs_are_not_clashes()
    test_relieve_clashes_rotates_the_variant_only()
    test_check_endpoints_raises_and_warns_on_a_clash()
    test_check_endpoints_silent_without_a_clash()
    test_minimize_removes_the_clash_and_passes_the_check()
    print("test_clash: all passed")
