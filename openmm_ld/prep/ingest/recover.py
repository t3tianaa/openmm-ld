"""Recover CHARMM tabulated LJ (NBFIX) after an OpenMM -> ParmEd conversion.

The lossy-direction rule: ParmEd -> OpenMM (createSystem) is lossless,but OpenMM -> ParmEd 
(load_topology) drops a CHARMM `CustomNonbondedForce` (the acoef/bcoef LJ table), so NBFIX is 
lost. Any code that crosses that boundary for a CHARMM system must recoverit. This module 
holds the recovery so every such site (ingest Mode-A CHARMM, solvation's water box) uses the 
SAME implementation.

`recover_charmm_lj` reads the tabulated force's A/B coefficients per atom TYPE pair
(E = A_ij/r^12 - B_ij/r^6): the DIAGONAL gives each atom's self LJ (sigma=(A/B)^(1/6),
eps=B^2/4A), and the OFF-DIAGONAL, wherever it deviates from Lorentz-Berthelot, is put back as a
ParmEd NBFIX -- so the ion-water NBFIX is preserved, not dropped. ParmEd then rebuilds the same
tabulated force on createSystem. Returns False for Amber, where LJ lives in the
NonbondedForce and there is no tabulated force to recover.
"""
from __future__ import annotations

import math

import openmm as mm
import parmed as pmd


def lj_from_coef(a_ij, b_ij):
    """(sigma_nm, eps_kj) from the A/r^12 - B/r^6 coefficients (A=acoef, B=bcoef)."""
    if a_ij <= 0 or b_ij <= 0:
        return None, 0.0
    return (a_ij / b_ij) ** (1.0 / 6.0), b_ij * b_ij / (4.0 * a_ij)


def recover_charmm_lj(struct, system) -> bool:
    """Recover the CHARMM tabulated LJ onto `struct`'s atom types. Returns True if it did.

    OpenMM units are nm/kJ; ParmEd stores Rmin (full, angstrom) and epsilon (kcal/mol).
    """
    cnbs = [f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce)]
    if not cnbs:
        return False
    cnb = cnbs[0]
    tabs = {cnb.getTabulatedFunctionName(t): cnb.getTabulatedFunction(t)
            for t in range(cnb.getNumTabulatedFunctions())}
    if "acoef" not in tabs or "bcoef" not in tabs:
        raise NotImplementedError(
            "force field uses a CustomNonbondedForce without acoef/bcoef tables; cannot "
            "recover its LJ. Use a ParmEd-native input (psf+params) or an amber14 water model.")
    an, _, av = tabs["acoef"].getFunctionParameters()
    bn, _, bv = tabs["bcoef"].getFunctionParameters()

    # 1. diagonal -> per-type self LJ
    diag: dict[int, tuple[float, float]] = {}
    type_at: dict[int, "pmd.AtomType"] = {}
    for i, atom in enumerate(struct.atoms):
        t = int(cnb.getParticleParameters(i)[0])
        if t not in type_at:
            sig_nm, eps_kj = lj_from_coef(av[t + an * t], bv[t + bn * t])
            diag[t] = (sig_nm if sig_nm else 0.1, eps_kj)
            rmin_half = (sig_nm or 0.1) * 10.0 * 2.0 ** (1.0 / 6.0) / 2.0   # nm -> Rmin/2 (A)
            at = pmd.AtomType(f"WT{t}", t + 1, atom.mass, atom.atomic_number)
            at.set_lj_params(eps_kj / 4.184, rmin_half)
            type_at[t] = at
        atom.atom_type = type_at[t]
        atom.type = type_at[t].name

    # 2. off-diagonal -> NBFIX wherever the pair deviates from Lorentz-Berthelot
    types = sorted(type_at)
    for x in range(len(types)):
        for y in range(x, len(types)):
            t1, t2 = types[x], types[y]
            sig_ij, eps_ij = lj_from_coef(av[t1 + an * t2], bv[t1 + bn * t2])
            if sig_ij is None:
                continue
            s1, e1 = diag[t1]
            s2, e2 = diag[t2]
            sig_lb = 0.5 * (s1 + s2)
            eps_lb = math.sqrt(e1 * e2)
            if (abs(sig_ij - sig_lb) > 1e-6 * max(1.0, sig_lb)
                    or abs(eps_ij - eps_lb) > 1e-6 * max(1.0, eps_lb)):
                rmin_full = sig_ij * 10.0 * 2.0 ** (1.0 / 6.0)             # nm -> Rmin (A)
                eps_kcal = eps_ij / 4.184
                type_at[t1].add_nbfix(type_at[t2].name, rmin_full, eps_kcal, rmin_full, eps_kcal)
                if t1 != t2:
                    type_at[t2].add_nbfix(type_at[t1].name, rmin_full, eps_kcal,
                                          rmin_full, eps_kcal)

    struct.unknown_functional = False
    return True
