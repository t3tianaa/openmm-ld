"""openmm_ld.prep against OpenMM-MSLD's prep on the small case. MDM2 itself: compare_prep.py."""
import os
import sys
import warnings

import pytest

import compare_prep
import openmm_ld.prep as new_prep

ENGINE = os.environ.get("MSLD_ENGINE_PATH")
pytestmark = pytest.mark.skipif(not ENGINE, reason="MSLD_ENGINE_PATH not set")
if ENGINE:
    sys.path.insert(0, ENGINE)


def test_peptide_build_is_identical():
    import msld.prep as old_prep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new, old = compare_prep.build_peptide(new_prep), compare_prep.build_peptide(old_prep)
    assert compare_prep.compare(new, old) == []


def test_raw_inputs_are_the_same_as_in_msld():
    import filecmp
    for name in os.listdir(compare_prep.MDM2):
        if name.endswith(".pdb"):
            assert filecmp.cmp(os.path.join(compare_prep.MDM2, name),
                               os.path.join(ENGINE, "test_systems", "mdm2", name), shallow=False), name
