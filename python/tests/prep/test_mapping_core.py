"""Phase A foundations: CoreMapping v2 + HybridBlueprint + the Mapper base.

Covers the pieces the prep redesign adds on top of the P1 data model:
  - CoreMapping carries the new v2 fields (core_corresp, atom_names, source_names) and
    validates them,
  - CoreMapping and HybridBlueprint round-trip through JSON exactly (block order and all),
  - HybridBlueprint.validate / global_env / to_model behave,
  - Mapper is the base interface; ManualMapper implements get_mapping() and is a Mapper instance.
"""
import os
import sys

from openmm_ld.prep import (CoreMapping, HybridBlueprint, Mapper,
                       ManualMapper, model_from_mappings)
from openmm_ld.model import MSLDModel


def _site1() -> CoreMapping:
    return CoreMapping(
        env_atoms={0, 1, 2, 3},
        blocks={"A": {4, 5}, "B": {6}},
        attachments=[(1, 4), (1, 6)],
        core_charges={0: -0.3, 1: 0.1},
        core_corresp={"A": {4: 1}, "B": {6: 1}},
        atom_names={4: "CB", 5: "HB", 6: "CB1"},
        source_names={4: "CB", 5: "HB", 6: "CB"},
    )


def _site2() -> CoreMapping:
    return CoreMapping(env_atoms={0, 1, 2, 3},
                       blocks={"C": {7, 8}, "D": {9, 10, 11}},
                       attachments=[(2, 7), (2, 9)])


# --------------------------------------------------------------- CoreMapping v2
def test_v2_fields_present_and_validate():
    m = _site1().validate()
    assert m.block_names == ["A", "B"]                     # insertion order
    assert m.all_block_atoms() == {4, 5, 6}
    assert m.core_corresp["A"] == {4: 1}
    assert m.atom_names[6] == "CB1" and m.source_names[6] == "CB"


def test_validate_rejects_corresp_unknown_block():
    m = CoreMapping(env_atoms={0}, blocks={"A": {1}, "B": {2}},
                    core_corresp={"Z": {1: 0}})            # 'Z' is not a block
    try:
        m.validate()
        assert False, "expected unknown-block error"
    except ValueError as ex:
        assert "core_corresp" in str(ex)


def test_json_round_trip_exact():
    m = _site1()
    back = CoreMapping.from_json(m.to_json())
    # sets compare equal regardless of order; block order must survive
    assert back.env_atoms == m.env_atoms
    assert back.blocks == m.blocks
    assert back.block_names == m.block_names               # order preserved
    assert back.attachments == m.attachments               # tuples restored
    assert back.core_charges == m.core_charges             # int keys restored
    assert back.core_corresp == m.core_corresp             # nested int keys restored
    assert back.atom_names == m.atom_names
    assert back.source_names == m.source_names
    back.validate()


def test_json_dict_keys_are_strings():
    # a JSON object cannot have int keys; to_dict must stringify them
    import json
    raw = json.loads(_site1().to_json())
    assert all(isinstance(k, str) for k in raw["core_charges"])
    assert all(isinstance(k, str) for k in raw["atom_names"])


# --------------------------------------------------------------- HybridBlueprint
def test_blueprint_validate_and_global_env():
    bp = HybridBlueprint(n_atoms=12, sites=[_site1(), _site2()]).validate()
    # env = everything not owned by a block: {0,1,2,3} here (4..11 are blocks)
    assert bp.global_env == {0, 1, 2, 3}


def test_blueprint_rejects_atom_in_two_sites():
    a = CoreMapping(env_atoms={0}, blocks={"A": {1}, "B": {2}})
    b = CoreMapping(env_atoms={0}, blocks={"C": {2}, "D": {3}})   # atom 2 reused
    try:
        HybridBlueprint(n_atoms=4, sites=[a, b]).validate()
        assert False, "expected atom-in-two-sites error"
    except ValueError as ex:
        assert "two sites" in str(ex)


def test_blueprint_rejects_atom_out_of_range():
    a = CoreMapping(env_atoms={0}, blocks={"A": {1}, "B": {9}})   # 9 >= n_atoms
    try:
        HybridBlueprint(n_atoms=4, sites=[a]).validate()
        assert False, "expected out-of-range error"
    except ValueError as ex:
        assert "n_atoms" in str(ex)


def test_blueprint_json_round_trip():
    bp = HybridBlueprint(n_atoms=12, sites=[_site1(), _site2()])
    back = HybridBlueprint.from_json(bp.to_json())
    assert back.n_atoms == 12
    assert [s.block_names for s in back.sites] == [["A", "B"], ["C", "D"]]
    back.validate()


def test_blueprint_to_model_matches_model_from_mappings():
    bp = HybridBlueprint(n_atoms=12, sites=[_site1(), _site2()])
    model = bp.to_model()
    assert isinstance(model, MSLDModel)
    assert model.n_blocks == 5 and model.n_sites == 3
    assert model.blocks_per_site == [1, 2, 2]
    # identical to the free-function path
    ref = model_from_mappings(12, [_site1(), _site2()])
    assert model.atom_block == ref.atom_block


# --------------------------------------------------------------- Mapper base
def test_manual_mapper_is_mapper_and_get_mapping_works():
    mm = ManualMapper(env_atoms={0, 1}, blocks={"A": {2, 3}, "B": {4}})
    assert isinstance(mm, Mapper)
    m = mm.get_mapping()
    assert m.block_names == ["A", "B"]
    m.validate()


def test_get_mapping_ignores_reference_and_variants():
    # a labeling mapper ignores reference/variants (they exist for discovery mappers)
    mm = ManualMapper(env_atoms={0}, blocks={"A": {1}, "B": {2}})
    m = mm.get_mapping(reference="ignored", variants={"A": "ignored"})
    assert m.all_block_atoms() == {1, 2}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nPhase A mapping-core tests passed")
