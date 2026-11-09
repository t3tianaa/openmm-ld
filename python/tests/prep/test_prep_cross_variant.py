"""Cross-variant bonded-term check / delete.

A bonded term connecting two blocks of the same site is illegal in MSLD. Our merge
never creates one, so this is a safety net for topologies built elsewhere: detect them,
report clearly, and optionally delete them before building.

Layout: env {0,1,2}, site 1 blocks A={3}, B={4}. We plant an illegal angle (3-2-4)
that spans A and B, then check we find it, delete it, and can build afterwards.
"""
import os
import sys

import numpy as np
from openmm import app

import parmed as pmd
from openmm_ld.prep import (ManualMapper, model_from_mappings,
                       find_cross_variant_terms, assert_no_cross_variant_terms,
                       delete_cross_variant_terms)
from openmm_ld.builder import scale_msld_system


def _structure(with_illegal_angle=True):
    """env 0,1,2 + block A=3 + block B=4; A and B both bond to env atom 2."""
    s = pmd.Structure()
    for i in range(5):
        a = pmd.Atom(name=f"A{i}", atomic_number=6, charge=0.0, mass=12.0, type=f"C{i}")
        at = pmd.AtomType(f"C{i}", i + 1, 12.0, 6); at.set_lj_params(0.1, 1.6)
        a.atom_type = at
        s.add_atom(a, "RES", 1)
    bt = pmd.BondType(300.0, 1.5); s.bond_types.append(bt)
    for i, j in [(0, 1), (1, 2), (2, 3), (2, 4)]:
        s.bonds.append(pmd.Bond(s.atoms[i], s.atoms[j], type=bt))
    angt = pmd.AngleType(50.0, 110.0); s.angle_types.append(angt)
    s.angles.append(pmd.Angle(s.atoms[0], s.atoms[1], s.atoms[2], type=angt))   # legal
    s.angles.append(pmd.Angle(s.atoms[1], s.atoms[2], s.atoms[3], type=angt))   # legal (env-A)
    if with_illegal_angle:
        s.angles.append(pmd.Angle(s.atoms[3], s.atoms[2], s.atoms[4], type=angt))  # A-env-B !!
    s.coordinates = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0],
                              [4.0, 1.0, 0], [4.0, -1.0, 0]], dtype=float)
    return s


def _mapping():
    return ManualMapper(env_atoms={0, 1, 2}, blocks={"A": {3}, "B": {4}}).get_mapping()


def test_find_and_assert_detect_cross_variant():
    s = _structure(with_illegal_angle=True)
    m = _mapping()
    found = find_cross_variant_terms(s, m)
    assert len(found) == 1, found
    cat, idxs, _term = found[0]
    assert cat == "angle" and set(idxs) == {2, 3, 4}
    try:
        assert_no_cross_variant_terms(s, m)
        assert False, "expected assert to raise"
    except ValueError as ex:
        assert "cross-variant" in str(ex)


def test_clean_structure_has_none():
    s = _structure(with_illegal_angle=False)
    m = _mapping()
    assert find_cross_variant_terms(s, m) == []
    assert_no_cross_variant_terms(s, m)              # no raise


def test_delete_then_build_succeeds():
    s = _structure(with_illegal_angle=True)
    m = _mapping()
    # before: scale_msld_system would reject the illegal angle
    n = delete_cross_variant_terms(s, m)
    assert n == 1
    assert find_cross_variant_terms(s, m) == []      # gone
    assert len(s.angles) == 2                         # only the two legal angles remain

    # now the engine accepts it
    system = s.createSystem(nonbondedMethod=app.NoCutoff)
    model = model_from_mappings(5, [m], add_restraints=False)
    info = scale_msld_system(system, model)           # must NOT raise
    assert "scaled_bond_forces" in info


def test_cross_site_term_is_legal():
    # a term spanning two DIFFERENT sites is allowed (product lambda), not flagged
    s = pmd.Structure()
    for i in range(5):
        a = pmd.Atom(name=f"A{i}", atomic_number=6, charge=0.0, mass=12.0, type=f"C{i}")
        at = pmd.AtomType(f"C{i}", i + 1, 12.0, 6); at.set_lj_params(0.1, 1.6)
        a.atom_type = at
        s.add_atom(a, "RES", 1)
    bt = pmd.BondType(300.0, 1.5); s.bond_types.append(bt)
    s.bonds.append(pmd.Bond(s.atoms[1], s.atoms[3], type=bt))    # site1-blockA to site2-blockC
    s.coordinates = np.zeros((5, 3))
    # site 1: A={1}, B={2} ; site 2: C={3}, D={4}
    site1 = ManualMapper(env_atoms={0}, blocks={"A": {1}, "B": {2}}).get_mapping()
    site2 = ManualMapper(env_atoms={0}, blocks={"C": {3}, "D": {4}}).get_mapping()
    assert find_cross_variant_terms(s, [site1, site2]) == []      # different sites -> legal


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nP3 cross-variant tests passed")
