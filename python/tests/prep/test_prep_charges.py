"""Charge policy (renormalize block charges + net-charge guard).

  env   = {0, 1}   with fixed charges  q0=-0.4, q1=+0.1  -> env_local sum = -0.30
  site 1: block "A" = {2, 3}, block "B" = {4}
"""
import os
import sys
import warnings

from openmm_ld.prep import (ManualMapper, assert_iso_charge, average_env_charges,
                       renormalize_charges, variant_net_charges)


def _mapping(buffer=None):
    return ManualMapper(env_atoms={0, 1}, blocks={"A": {2, 3}, "B": {4}},
                        buffer_atoms=(buffer or set())).get_mapping()


# net charges
def test_variant_net_charges():
    m = _mapping()
    q = {0: -0.4, 1: 0.1, 2: 0.2, 3: 0.05, 4: 0.31}
    nets = variant_net_charges(m, q)
    assert abs(nets["A"] - (-0.30 + 0.25)) < 1e-12      # -0.05
    assert abs(nets["B"] - (-0.30 + 0.31)) < 1e-12      #  0.01


def test_missing_charge_raises():
    m = _mapping()
    try:
        variant_net_charges(m, {0: -0.4, 1: 0.1, 2: 0.2})   # 3 and 4 missing
        assert False, "expected missing-charge error"
    except ValueError as ex:
        assert "missing" in str(ex)


# renormalize
def test_renormalize_hits_integer_and_leaves_env():
    m = _mapping()
    q = {0: -0.4, 1: 0.1, 2: 0.2, 3: 0.05, 4: 0.31}     # both variants -> target 0
    r = renormalize_charges(m, q)

    # env untouched
    assert r.charges[0] == -0.4 and r.charges[1] == 0.1
    # both variants land exactly on 0
    assert abs(r.final_net["A"]) < 1e-12
    assert abs(r.final_net["B"]) < 1e-12
    assert r.target == {"A": 0, "B": 0}
    # correction spread EQUALLY over block A's two atoms (+0.05 total -> +0.025 each)
    assert abs(r.correction["A"] - 0.05) < 1e-12
    assert abs(r.charges[2] - 0.225) < 1e-12
    assert abs(r.charges[3] - 0.075) < 1e-12
    # block B single atom absorbs its whole correction (-0.01)
    assert abs(r.charges[4] - 0.30) < 1e-12
    # input dict not mutated
    assert q[2] == 0.2


def test_proportional_method():
    m = _mapping()
    q = {0: -0.4, 1: 0.1, 2: 0.2, 3: 0.05, 4: 0.31}
    r = renormalize_charges(m, q, method="proportional")
    # correction +0.05 split by |charge|: 0.2 and 0.05 -> shares 0.04 and 0.01
    assert abs(r.charges[2] - (0.2 + 0.04)) < 1e-12
    assert abs(r.charges[3] - (0.05 + 0.01)) < 1e-12
    assert abs(r.final_net["A"]) < 1e-12                 # still lands on 0


def test_guard_rejects_charge_change():
    m = _mapping()
    # make block B a +1 residue: raw net ~ +1, block A ~ 0  -> different targets
    q = {0: -0.4, 1: 0.1, 2: 0.2, 3: 0.05, 4: 1.31}
    try:
        renormalize_charges(m, q)
        assert False, "expected net-charge guard to fire"
    except ValueError as ex:
        assert "different net charges" in str(ex)


def test_assert_iso_charge_returns_shared_integer():
    m = _mapping()
    q = {0: -0.4, 1: 0.1, 2: 0.2, 3: 0.05, 4: 0.31}
    assert assert_iso_charge(m, q) == 0


def test_far_from_integer_needs_explicit_target():
    m = _mapping()
    # block A raw net = -0.30 + (0.4+0.2) = +0.30, far from any integer
    q = {0: -0.4, 1: 0.1, 2: 0.4, 3: 0.2, 4: 0.31}
    try:
        renormalize_charges(m, q)
        assert False, "expected far-from-integer error"
    except ValueError as ex:
        assert "far from an integer" in str(ex)


def test_explicit_targets_used():
    m = _mapping()
    # both variants forced to +1 (same integer -> guard passes)
    q = {0: -0.4, 1: 0.1, 2: 0.4, 3: 0.2, 4: 1.31}
    r = renormalize_charges(m, q, targets={"A": 1, "B": 1})
    assert r.target == {"A": 1, "B": 1}
    assert abs(r.final_net["A"] - 1.0) < 1e-12
    assert abs(r.final_net["B"] - 1.0) < 1e-12


def test_buffer_not_implemented():
    m = _mapping(buffer={0})
    q = {0: -0.4, 1: 0.1, 2: 0.2, 3: 0.05, 4: 0.31}
    try:
        renormalize_charges(m, q)
        assert False, "expected buffer NotImplementedError"
    except NotImplementedError as ex:
        assert "buffer" in str(ex)


# average env charges across variants (Amber backbone case)
def test_average_env_identical_is_noop():
    # CHARMM-like: every variant has the same backbone charges -> average = same, silent
    per_variant = [[-0.4157, 0.2719, 0.07], [-0.4157, 0.2719, 0.07]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = average_env_charges(per_variant, labels=["N", "H", "CA"])
    assert rep.averaged == [-0.4157, 0.2719, 0.07]
    assert rep.differing == []
    assert rep.max_spread < 1e-9
    assert len(caught) == 0                       # no message when nothing differs


def test_average_env_differing_takes_mean_and_warns():
    # Amber-like: CA differs (0.10 vs -0.04), rest identical
    per_variant = [[-0.4157, 0.2719, 0.10], [-0.4157, 0.2719, -0.04]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = average_env_charges(per_variant, labels=["N", "H", "CA"])
    assert abs(rep.averaged[2] - 0.03) < 1e-12    # (0.10 + -0.04)/2
    assert rep.averaged[0] == -0.4157             # unchanged atoms untouched
    assert len(rep.differing) == 1 and rep.differing[0][0] == "CA"
    assert abs(rep.max_spread - 0.14) < 1e-12
    assert len(caught) == 1 and "average" in str(caught[0].message)


def test_average_env_refuses_large_spread():
    # a huge spread -> not a shared atom (backbone mutation) -> refuse
    per_variant = [[0.0, 0.1], [0.0, 1.0]]
    try:
        average_env_charges(per_variant, labels=["N", "X"])
        assert False, "expected refuse on large spread"
    except ValueError as ex:
        assert "not really a shared atom" in str(ex)


def test_average_env_length_mismatch_raises():
    try:
        average_env_charges([[0.1, 0.2], [0.1]])
        assert False, "expected length-mismatch error"
    except ValueError as ex:
        assert "correspond" in str(ex)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nP1.5 prep-charges tests passed")
