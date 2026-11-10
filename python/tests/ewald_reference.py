"""A plain Ewald sum of the MSLD nonbonded energy, written from the energy spec with no OpenMM code.

Independent reference for the plugin's PME at partial lambda (the NumPy oracle only covers NoCutoff).
The reciprocal part is the direct k-space sum, converged by `k_tol`, with no grid. Pairs follow the
spec (vault: MSLD energy function spec):

  direct space   L KE q_i q_j g(r_sc) + L 4 eps (1/s^2 - 1/s), r < cutoff, g(x) = erfc(alpha x)/x
  1-4 pairs      L (KE qq14/r + 4 eps14 ((s14/r)^12 - (s14/r)^6)), no cutoff, no soft-core
  reciprocal     KE (2 pi/V) sum_{k != 0} exp(-k^2/(4 alpha^2))/k^2 |S(k)|^2, S = sum_j Q_j exp(i k.r_j),
                 with Q_j = lambda_b(j) q_j
  self           -KE alpha/sqrt(pi) sum_j Q_j^2
  excluded pairs -KE Q_i Q_j erf(alpha r)/r, for every exception and same-site cross-block pair
  background     -KE pi (sum_j Q_j)^2 / (2 V alpha^2), for a net charge (as NonbondedForce)

Every function is complex-analytic, so dU/dlambda comes from a complex step of this same energy and
there are no hand-written derivatives here. Units: nm, kJ/mol, e. The box is cubic.
"""
import itertools
import math

import numpy as np
from scipy.special import erf, erfc


def _coupling(lam, bi, bj):
    if bi == 0 and bj == 0:
        return None
    if bi == bj or bi == 0 or bj == 0:
        return lam[max(bi, bj)]
    return lam[bi] * lam[bj]


def reciprocal(x, Q, alpha, box, ke, k_tol=1e-18):
    """The k-space sum over half of k-space, doubled. Stops where the Gaussian factor falls below k_tol."""
    kmax = 2.0 * alpha * math.sqrt(-math.log(k_tol))
    nmax = int(math.ceil(kmax * box / (2.0 * math.pi)))
    grid = np.array([n for n in itertools.product(range(-nmax, nmax + 1), repeat=3)
                     if n > (0, 0, 0) and sum(c * c for c in n) * (2 * math.pi / box) ** 2 <= kmax * kmax], dtype=float)
    k = grid * (2.0 * math.pi / box)
    k2 = np.sum(k * k, axis=1)
    weight = np.exp(-k2 / (4.0 * alpha * alpha)) / k2
    phase = k @ x.T
    c, s = np.cos(phase) @ Q, np.sin(phase) @ Q
    return ke * (4.0 * math.pi / box ** 3) * np.sum(weight * (c * c + s * s))


def energy(spec, lam, alpha, box, ke, cutoff, lj=True, images=0, k_tol=1e-18):
    """Total nonbonded energy at lambda (per block, index 0 = environment).

    images = 0: minimum image, direct space cut at `cutoff`, as the spec and the plugin do.
    images = n: direct space over all images up to +-n boxes with no cutoff (only for Coulomb
    without soft-core, to test this reference itself).
    """
    x = np.asarray(spec["positions"], dtype=float)
    q, sig, eps = spec["charge"], spec["sigma"], spec["epsilon"]
    block, site = spec["atom_block"], spec["block_site"]
    sc_alpha, sq6 = spec["alpha"], spec["sigma_q"] ** 6
    n = len(x)
    exceptions = {(min(i, j), max(i, j)) for i, j in spec["excluded"]}
    excluded = set(exceptions)
    for i in range(n):
        for j in range(i + 1, n):
            if block[i] and block[j] and block[i] != block[j] and site[block[i]] == site[block[j]]:
                excluded.add((i, j))
    Q = np.array([lam[block[i]] * q[i] for i in range(n)])
    U = 0.0

    # direct space
    if images:
        shifts = [np.array(t, dtype=float) * box for t in itertools.product(range(-images, images + 1), repeat=3)]
    for i in range(n):
        for j in range(i, n):
            pair_excluded = (i, j) in excluded
            if images:
                for shift in shifts:
                    is_zero = not np.any(shift)
                    if (is_zero and (i == j or pair_excluded)):
                        continue
                    r = np.linalg.norm(x[j] + shift - x[i])
                    # i == j counts each image pair twice
                    U = U + (0.5 if i == j else 1.0) * ke * Q[i] * Q[j] * erfc(alpha * r) / r
                continue
            if i == j or pair_excluded:
                continue
            d = x[j] - x[i]
            d -= box * np.round(d / box)
            r = float(np.linalg.norm(d))
            if r >= cutoff:
                continue
            L = _coupling(lam, block[i], block[j])
            shift = 0.0 if L is None else sc_alpha * (1.0 - L)
            L = 1.0 if L is None else L
            rsc = (r ** 6 + shift * sq6) ** (1.0 / 6.0)
            U = U + L * ke * q[i] * q[j] * erfc(alpha * rsc) / rsc
            if lj and eps[i] != 0.0 and eps[j] != 0.0:
                sg, ep = 0.5 * (sig[i] + sig[j]), math.sqrt(eps[i] * eps[j])
                s = (r / sg) ** 6 + shift
                U = U + L * 4.0 * ep * (1.0 / (s * s) - 1.0 / s)

    # 1-4 pairs (not periodic, as NonbondedForce by default)
    for i, j, qq, s14, e14 in spec["pairs14"]:
        r = float(np.linalg.norm(x[j] - x[i]))
        L = _coupling(lam, block[i], block[j])
        u = ke * qq / r
        if lj:
            u = u + 4.0 * e14 * ((s14 / r) ** 12 - (s14 / r) ** 6)
        U = U + (u if L is None else L * u)

    # reciprocal space, self term, excluded pairs, net-charge background
    U = U + reciprocal(x, Q, alpha, box, ke, k_tol)
    U = U - ke * alpha / math.sqrt(math.pi) * np.sum(Q * Q)
    for i, j in excluded:
        d = x[j] - x[i]
        d -= box * np.round(d / box)
        r = float(np.linalg.norm(d))
        U = U - ke * Q[i] * Q[j] * erf(alpha * r) / r
    total = np.sum(Q)
    U = U - ke * math.pi * total * total / (2.0 * box ** 3 * alpha * alpha)
    return U


def denergy_dlambda(spec, lam, alpha, box, ke, cutoff, lj=True, k_tol=1e-18, h=1e-30):
    """dU/dlambda_b for b >= 1 by complex step (exact to rounding)."""
    out = np.zeros(len(lam))
    for b in range(1, len(lam)):
        lc = np.array(lam, dtype=complex)
        lc[b] += 1j * h
        out[b] = np.imag(energy(spec, lc, alpha, box, ke, cutoff, lj=lj, k_tol=k_tol)) / h
    return out
