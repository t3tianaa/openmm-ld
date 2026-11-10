"""Phase B: AminoAcidBackboneMapper (name+type common-core mapping).

The single rule (vault: Mapping component): env = the maximal core shared by NAME *and*
TYPE across all variants; everything else is a block; a backbone heavy atom (N/CA/C/O) in a
block flags a backbone mutation. Verified behaviors:

  - Amber-style uniform types: full backbone (and any type-matched side-chain atom, e.g. CB)
    is shared env; no backbone mutation.
  - "bigger core": Phe/Tyr share the ring up to the atom whose TYPE diverges (CZ), which
    becomes a block -> env extends beyond the backbone.
  - CHARMM Gly: CA type differs (CT1 vs CT2) -> CA drops into the blocks -> backbone mutation.

Toy residues are built with ParmEd (the real substrate) but need no force field: the mapper
reads only atom .name / .type / .idx.
"""
import os
import sys

import parmed as pmd
from openmm_ld.prep import AminoAcidBackboneMapper, CoreMapping, Mapper

_Z = {"N": 7, "C": 6, "O": 8, "H": 1, "S": 16}


def residue(resname, atoms):
    """atoms = list of (atom_name, atom_type); element inferred from the name's first letter."""
    st = pmd.Structure()
    for an, at in atoms:
        a = pmd.Atom(name=an, type=at, atomic_number=_Z.get(an[0], 6))
        st.add_atom(a, resname, 1)
    return st


# Amber-style backbone: uniform types across residues (verified: ff14SB N/CA/C/O/H/HA identical)
_BB_AMBER = [("N", "N"), ("CA", "CX"), ("C", "C"), ("O", "O"), ("HN", "H"), ("HA", "H1")]


def _env_names(m: CoreMapping):
    return sorted(m.atom_names[i] for i in m.env_atoms)


def _block_src(m: CoreMapping, name):
    return sorted(m.source_names[i] for i in m.blocks[name])


# --------------------------------------------------------------- side-chain mutation (Amber)
def test_sidechain_mutation_shares_backbone_and_cb():
    ala = residue("ALA", _BB_AMBER + [("CB", "CT"), ("HB1", "HC"), ("HB2", "HC"), ("HB3", "HC")])
    val = residue("VAL", _BB_AMBER + [("CB", "CT"), ("CG1", "CT"), ("HG11", "HC"), ("CG2", "CT")])
    m = AminoAcidBackboneMapper(reference_name="ALA").get_mapping(ala, {"VAL": val})

    # CB is type-matched (CT) in both -> it joins the env (bigger core than just backbone)
    assert _env_names(m) == sorted(["N", "CA", "C", "O", "HN", "HA", "CB"])
    assert m.metadata["backbone_mutation"] is False
    # blocks are the non-shared remainder
    assert _block_src(m, "ALA") == ["HB1", "HB2", "HB3"]
    assert _block_src(m, "VAL") == ["CG1", "CG2", "HG11"]
    # both blocks + env partition all atoms, indices unique
    m.validate()


def test_reference_competes_by_default():
    ala = residue("ALA", _BB_AMBER + [("CB", "CT"), ("HB1", "HC")])
    val = residue("VAL", _BB_AMBER + [("CB", "CT"), ("CG1", "CT")])
    m = AminoAcidBackboneMapper(reference_name="ALA").get_mapping(ala, {"VAL": val})
    assert m.block_names == ["ALA", "VAL"]           # WT included, first (reference frame)


def test_include_reference_false_needs_two_variants():
    ala = residue("ALA", _BB_AMBER + [("CB", "CT")])
    val = residue("VAL", _BB_AMBER + [("CB", "CT"), ("CG1", "CT")])
    try:
        AminoAcidBackboneMapper(include_reference=False).get_mapping(ala, {"VAL": val})
        assert False, "expected a >=2-variant error"
    except ValueError as ex:
        assert "2 competing" in str(ex)


# --------------------------------------------------------------- bigger core (Phe / Tyr)
def test_phe_tyr_share_ring_up_to_type_boundary():
    ring = [("CB", "CT"), ("CG", "CA"), ("CD1", "CA"), ("CD2", "CA"),
            ("CE1", "CA"), ("CE2", "CA")]
    phe = residue("PHE", _BB_AMBER + ring + [("CZ", "CA"), ("HZ", "HA")])
    tyr = residue("TYR", _BB_AMBER + ring + [("CZ", "CO"), ("OH", "OH"), ("HH", "HO")])
    m = AminoAcidBackboneMapper(reference_name="PHE").get_mapping(phe, {"TYR": tyr})

    # env extends through the ring but STOPS at CZ (type CA vs CO) -> CZ is a block atom
    env = set(_env_names(m))
    assert {"CB", "CG", "CD1", "CD2", "CE1", "CE2"} <= env
    assert "CZ" not in env
    assert m.metadata["backbone_mutation"] is False
    assert _block_src(m, "PHE") == ["CZ", "HZ"]
    assert _block_src(m, "TYR") == ["CZ", "HH", "OH"]


# --------------------------------------------------------------- backbone mutation (CHARMM Gly)
def test_charmm_glycine_is_a_backbone_mutation():
    # CHARMM types: Ala CA=CT1, Gly CA=CT2 -> CA differs in type -> not shared
    ala = residue("ALA", [("N", "NH1"), ("CA", "CT1"), ("C", "C"), ("O", "O"),
                          ("HN", "H"), ("HA", "HB1"), ("CB", "CT3"), ("HB", "HA3")])
    gly = residue("GLY", [("N", "NH1"), ("CA", "CT2"), ("C", "C"), ("O", "O"),
                          ("HN", "H"), ("HA2", "HB2"), ("HA3", "HB2")])
    m = AminoAcidBackboneMapper(reference_name="ALA").get_mapping(ala, {"GLY": gly})

    assert m.metadata["backbone_mutation"] is True
    assert m.metadata["mutated_backbone_atoms"] == ["CA"]
    # env is only the type-matched backbone atoms; CA is NOT in env
    assert set(_env_names(m)) == {"N", "C", "O", "HN"}
    # CA appears (as a block atom) in BOTH competing blocks
    assert "CA" in _block_src(m, "ALA") and "CA" in _block_src(m, "GLY")


# --------------------------------------------------------------- correspondence & index maps
def test_variant_index_maps_share_env_indices():
    ala = residue("ALA", _BB_AMBER + [("CB", "CT"), ("HB1", "HC")])
    val = residue("VAL", _BB_AMBER + [("CB", "CT"), ("CG1", "CT")])
    m = AminoAcidBackboneMapper(reference_name="ALA").get_mapping(ala, {"VAL": val})

    # the shared atom N maps to the SAME hybrid index from every variant
    def hybrid_of(struct, mapname, atomname):
        local = next(a.idx for a in struct.atoms if a.name == atomname)
        return m.variant_index_maps[mapname][local]
    assert hybrid_of(ala, "ALA", "N") == hybrid_of(val, "VAL", "N")
    assert hybrid_of(ala, "ALA", "CB") == hybrid_of(val, "VAL", "CB")
    # every local atom of each variant is mapped
    assert len(m.variant_index_maps["ALA"]) == len(ala.atoms)
    assert len(m.variant_index_maps["VAL"]) == len(val.atoms)


def test_name_collision_renamed_unique():
    # both blocks own an atom literally named "X1" (different type -> both are block atoms)
    a = residue("A", _BB_AMBER + [("X1", "ta")])
    b = residue("B", _BB_AMBER + [("X1", "tb")])
    m = AminoAcidBackboneMapper(reference_name="A").get_mapping(a, {"B": b})
    names = list(m.atom_names.values())
    assert len(names) == len(set(names)), "hybrid atom names must be unique"
    # originals preserved in source_names
    srcs = [m.source_names[i] for blk in m.blocks.values() for i in blk]
    assert srcs.count("X1") == 2


# --------------------------------------------------------------- plumbing
def test_is_a_mapper_and_json_round_trips():
    ala = residue("ALA", _BB_AMBER + [("CB", "CT"), ("HB1", "HC")])
    val = residue("VAL", _BB_AMBER + [("CB", "CT"), ("CG1", "CT")])
    mapper = AminoAcidBackboneMapper(reference_name="ALA")
    assert isinstance(mapper, Mapper)
    m = mapper.get_mapping(ala, {"VAL": val})
    back = CoreMapping.from_json(m.to_json())
    assert back.blocks == m.blocks
    assert back.variant_index_maps == m.variant_index_maps
    assert back.metadata == m.metadata
    back.validate()


def test_requires_reference():
    try:
        AminoAcidBackboneMapper().get_mapping(None, {"A": residue("A", _BB_AMBER)})
        assert False, "expected a missing-reference error"
    except ValueError as ex:
        assert "reference" in str(ex)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nPhase B backbone-mapper tests passed")
