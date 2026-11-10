"""Prep data model (CoreMapping + ManualMapper + model_from_mappings).

  - a CoreMapping validates (and rejects broken input),
  - ManualMapper produces a valid CoreMapping,
  - model_from_mappings builds an MSLDModel whose sites/blocks/atom_block match the
    rules in model.py, and adds one CATS restraint per (multi-atom) block.

Toy layout used (12 atoms):
  atoms 0-3   = environment (shared backbone)
  site 1: block "A" = {4,5}, block "B" = {6}
  site 2: block "C" = {7,8}, block "D" = {9,10,11}
"""
import os
import sys

from openmm_ld.prep import ManualMapper, model_from_mappings


def _site1():
    return ManualMapper(
        env_atoms={0, 1, 2, 3},
        blocks={"A": {4, 5}, "B": {6}},
        attachments=[(1, 4), (1, 6)],          # both variants attach to env atom 1
    ).get_mapping()


def _site2():
    return ManualMapper(
        env_atoms={0, 1, 2, 3},
        blocks={"C": {7, 8}, "D": {9, 10, 11}},
        attachments=[(2, 7), (2, 9)],
    ).get_mapping()


# --------------------------------------------------------------------- CoreMapping
def test_mapping_validates_ok():
    m = _site1()
    assert m.block_names == ["A", "B"]               # insertion order preserved
    assert m.all_block_atoms() == {4, 5, 6}
    m.validate()                                     # no raise


def test_mapping_rejects_atom_in_two_blocks():
    try:
        ManualMapper(env_atoms={0}, blocks={"A": {4, 5}, "B": {5, 6}}).get_mapping()
        assert False, "expected overlap error"
    except ValueError as ex:
        assert "two blocks" in str(ex)


def test_mapping_rejects_env_block_overlap():
    try:
        ManualMapper(env_atoms={0, 4}, blocks={"A": {4, 5}, "B": {6}}).get_mapping()
        assert False, "expected env/block overlap error"
    except ValueError as ex:
        assert "environment" in str(ex)


def test_mapping_rejects_bad_attachment():
    try:
        ManualMapper(env_atoms={0, 1}, blocks={"A": {4}, "B": {5}},
                     attachments=[(1, 99)]).get_mapping()   # 99 is not a block atom
        assert False, "expected bad-attachment error"
    except ValueError as ex:
        assert "attachment" in str(ex)


def test_mapping_rejects_buffer_outside_env():
    try:
        ManualMapper(env_atoms={0, 1}, blocks={"A": {4}, "B": {5}},
                     buffer_atoms={2}).get_mapping()        # 2 is not in env
        assert False, "expected buffer error"
    except ValueError as ex:
        assert "buffer" in str(ex)


# --------------------------------------------------------------- model_from_mappings
def test_single_site_model():
    model = model_from_mappings(12, [_site1()], add_restraints=True)
    assert model.n_blocks == 3                       # env + A + B
    assert model.n_sites == 2                        # site 0 (env) + site 1
    assert model.blocks_per_site == [1, 2]
    assert model.site_bound == [0, 1, 3]
    assert model.blocks_in_site(1) == [1, 2]
    # atom->block assignment
    assert model.atom_block[4] == 1 and model.atom_block[5] == 1   # block A
    assert model.atom_block[6] == 2                                # block B
    assert model.atom_block[0] == 0                                # env
    # one CATS restraint for block A (>=2 atoms); block B has 1 atom -> skipped
    assert model.atom_restraints == [[4, 5]]


def test_two_site_model():
    model = model_from_mappings(12, [_site1(), _site2()], add_restraints=True)
    assert model.n_blocks == 5                       # env + A,B (site1) + C,D (site2)
    assert model.n_sites == 3
    assert model.blocks_per_site == [1, 2, 2]
    assert model.site_bound == [0, 1, 3, 5]
    assert model.blocks_in_site(1) == [1, 2]
    assert model.blocks_in_site(2) == [3, 4]
    assert model.atom_block[9] == 4 and model.atom_block[11] == 4  # block D
    # CATS restraints: A={4,5}, C={7,8}, D={9,10,11}; B is single-atom -> skipped
    assert model.atom_restraints == [[4, 5], [7, 8], [9, 10, 11]]
    # off by default (ISS-001 / ADR-0033): per-block centroid groups pull a switched-off side
    # chain onto itself
    assert model_from_mappings(12, [_site1(), _site2()]).atom_restraints == []


def test_block_options_passthrough():
    model = model_from_mappings(
        12, [_site1()],
        block_options={"A": {"theta0": 0.3, "theta_mass": 12.0}})
    a = model.blocks[1]
    assert a.site == 1 and abs(a.theta0 - 0.3) < 1e-12 and abs(a.theta_mass - 12.0) < 1e-12


def test_accepts_single_mapping_not_in_list():
    # passing one CoreMapping directly (not wrapped in a list) should also work
    model = model_from_mappings(12, _site1())
    assert model.n_sites == 2


def test_model_rejects_single_nonfixed_block_site():
    # a site with only one (non-fixed) block violates MSLDModel.finalize()
    one = ManualMapper(env_atoms={0}, blocks={"A": {4, 5}}).get_mapping()
    try:
        model_from_mappings(12, [one])
        assert False, "expected single-block site error"
    except ValueError as ex:
        assert "site" in str(ex)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nP1 prep-mapping tests passed")
