"""Stock-OpenMM MSLD nonbonded forces, for GPU platforms (CUDA, OpenCL) until the plugin has GPU kernels.

Vendored from OpenMM-MSLD msld/forces.py (sync branch b32a7fd), where this stack was validated: the plugin's
MSLDNonbondedForce was written from it and matches it to 1e-10 in energy and 1e-8 in dU/dlambda (oracle cases
and MDM2). Everything is analytic (CustomNonbondedForce / CustomBondForce parameter derivatives) except the
charges in the NonbondedForce (PME reciprocal + real space of lambda-scaled charges, linear 1-4 offsets). That
energy is an exact quadratic polynomial in the lambdas, so MSLDIntegrator differentiates it with a central
difference inside the step (exact for any step), on its own force group: see `apply_stock_nonbonded`.
"""
from __future__ import annotations

import math
from collections import defaultdict

import openmm as mm
from openmm import unit

from .forces import _forces_of_type, _lambda_prefix, _register_lambdas, lambda_name

_NM = unit.nanometer
_KJ = unit.kilojoule_per_mole
_E = unit.elementary_charge
_KE = 138.935456           # OpenMM ONE_4PI_EPS0 (kJ/mol * nm / e^2)
FD_FORCE_GROUP = 31        # the NonbondedForce: dU/dlambda by central difference in the integrator


def _alch_blocks(model) -> list[int]:
    """Non-environment blocks that actually own atoms."""
    return sorted({model.atom_block[a] for a in range(model.n_atoms)
                   if model.atom_block[a] != 0})


def _existing_exception_pairs(nbf: mm.NonbondedForce) -> set:
    pairs = set()
    for k in range(nbf.getNumExceptions()):
        p1, p2, *_ = nbf.getExceptionParameters(k)
        pairs.add((min(p1, p2), max(p1, p2)))
    return pairs


def _ewald_alpha(nbf: mm.NonbondedForce) -> float:
    """Ewald splitting parameter alpha (1/nm) the NonbondedForce will actually use.

    A user-set alpha (setPMEParameters, PME/LJPME only) wins; otherwise OpenMM derives it from
    the tolerance as (1/rc) * sqrt(-ln(2*tol)). Any real-space correction must use this same
    alpha, or partial-lambda energies are off (lambda = 1 stays exact). See ISS-008.
    """
    NB = mm.NonbondedForce
    if nbf.getNonbondedMethod() in (NB.PME, NB.LJPME):
        alpha = nbf.getPMEParameters()[0].value_in_unit(_NM**-1)
        if alpha > 0.0:
            return alpha
    rc = nbf.getCutoffDistance().value_in_unit(_NM)
    return math.sqrt(-math.log(2.0 * nbf.getEwaldErrorTolerance())) / rc


def _same_site_pairs(model) -> set:
    """All (a, c) atom pairs (a<c) whose atoms are in DIFFERENT blocks of the SAME site."""
    pairs = set()
    for s in range(model.n_sites):
        blocks = model.blocks_in_site(s)
        for x in range(len(blocks)):
            for y in range(x + 1, len(blocks)):
                for a in model.blocks[blocks[x]].atoms:
                    for c in model.blocks[blocks[y]].atoms:
                        pairs.add((min(a, c), max(a, c)))
    return pairs


def add_same_site_exclusions(nbf: mm.NonbondedForce, model) -> int:
    """Fully exclude every pair of atoms in different blocks of the SAME site
    (mutually-exclusive overlapping substituents must never interact). Skips
    pairs that already have an exception. Returns the number added."""
    existing = _existing_exception_pairs(nbf)
    added = 0
    for key in _same_site_pairs(model):
        if key in existing:
            continue
        nbf.addException(key[0], key[1], 0.0, 0.0, 0.0)  # full exclusion
        existing.add(key)
        added += 1
    return added


# ---------------------------------------------------------------------
# Within-block electrostatics correction (match BLaDE: linear, not lambda^2)
# ---------------------------------------------------------------------
# The charge-offset method scales each alchemical atom's charge by lambda_b, so a pair
# with BOTH atoms in block b gets lambda_b^2. That is correct for env<->block and
# cross-site pairs, but WRONG for within-block full pairs (>=1-5 apart): BLaDE scales
# them LINEARLY (nbdirect.cu uses lambda_b, not lambda_b^2, for same-block pairs). The
# two atoms of one substituent are perfectly correlated (present together with weight
# lambda_b), so their interaction should scale by lambda_b, not lambda_b^2.
#
# We cannot change the reciprocal (PME grid) per pair -- there BLaDE also gets lambda_b^2
# (scaled charges), and so do we; that matches. We correct only the DIRECT/real space:
# add (lambda_b - lambda_b^2) * [real-space Coulomb] for within-block full pairs. This is
# zero at lambda_b=1 (so lambda=1 == base) and turns the direct-space scaling from
# lambda_b^2 into lambda_b, exactly matching BLaDE. Only blocks with a 1-5+ internal pair
# get a force (small substituents are unaffected).


def add_within_block_elec_correction(system: mm.System, model) -> list:
    """Correct within-block electrostatics from lambda^2 to lambda in the direct space.

    MUST run BEFORE add_scaled_electrostatics (it reads the raw, unscaled charges).
    """
    created = []
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    if not nbfs:
        return created
    nbf = nbfs[0]
    npart = system.getNumParticles()
    qraw = [nbf.getParticleParameters(i)[0].value_in_unit(_E) for i in range(npart)]
    excl = sorted(_existing_exception_pairs(nbf))
    method = nbf.getNonbondedMethod()

    # direct-space Coulomb form + CustomNonbondedForce method. It must be the NonbondedForce's
    # own pair kernel g(r): erfc screen for Ewald/PME, reaction field (with its constant shift,
    # which does not cancel here) for CutoffPeriodic and CutoffNonPeriodic.
    NB, CN = mm.NonbondedForce, mm.CustomNonbondedForce
    if method == NB.NoCutoff:
        screen, cn_method, cutoff = "1/r", CN.NoCutoff, None
    elif method in (NB.PME, NB.Ewald, NB.LJPME):
        rc = nbf.getCutoffDistance().value_in_unit(_NM)
        alpha = _ewald_alpha(nbf)
        screen, cn_method, cutoff = f"erfc({alpha}*r)/r", CN.CutoffPeriodic, rc
    else:  # reaction field: CutoffPeriodic / CutoffNonPeriodic
        rc = nbf.getCutoffDistance().value_in_unit(_NM)
        eps = nbf.getReactionFieldDielectric()
        krf = (eps - 1.0) / ((2.0 * eps + 1.0) * rc ** 3)
        crf = 3.0 * eps / ((2.0 * eps + 1.0) * rc)
        screen = f"(1/r+{krf!r}*r^2-{crf!r})"
        cn_method = CN.CutoffPeriodic if method == NB.CutoffPeriodic else CN.CutoffNonPeriodic
        cutoff = rc

    excl_set = set(excl)
    # Same-site cross-block pairs are excluded on the base NonbondedForce (and LJ forces) but by
    # add_scaled_electrostatics, which runs AFTER this. The optimized platforms (CPU/OpenCL/CUDA)
    # require every nonbonded force to carry the SAME exclusion set, so add them here too. They are
    # cross-block and never in this force's within-block (atoms, atoms) group, so they contribute
    # zero energy -- this only harmonizes the exclusion lists so those platforms accept the System.
    same_site = _same_site_pairs(model)
    for b in _alch_blocks(model):
        atoms = list(model.blocks[b].atoms)
        # only needed if the block has a 1-5+ internal pair (not 1-2/1-3/1-4 excluded);
        # small/compact blocks whose pairs are all excluded need no correction force
        has_full_pair = any(
            (min(atoms[ii], atoms[jj]), max(atoms[ii], atoms[jj])) not in excl_set
            for ii in range(len(atoms)) for jj in range(ii + 1, len(atoms)))
        if not has_full_pair:
            continue
        lam = lambda_name(b)
        f = mm.CustomNonbondedForce(
            f"(lam - lam*lam)*{_KE}*q1*q2*({screen}); lam={lam}")
        f.setName("MSLDWBElec_%d" % b)
        f.addPerParticleParameter("q")
        for qi in qraw:
            f.addParticle([qi])
        for a, c in excl:               # skip 1-2/1-3/1-4 (correct only 1-5+ full pairs)
            f.addExclusion(a, c)
        for a, c in same_site:          # match the base/LJ forces' exclusions (platform requirement)
            if (a, c) not in excl_set:
                f.addExclusion(a, c)
        f.addGlobalParameter(lam, 1.0)
        f.addEnergyParameterDerivative(lam)
        f.setNonbondedMethod(cn_method)
        if cutoff is not None:
            f.setCutoffDistance(cutoff * _NM)
        f.setUseLongRangeCorrection(False)
        f.addInteractionGroup(atoms, atoms)
        system.addForce(f)
        created.append(f)
    return created


def add_scaled_electrostatics(system: mm.System, model) -> mm.NonbondedForce | None:
    """Scale each alchemical atom's charge to lambda_b * q_i via NonbondedForce
    parameter offsets, and add same-site exclusions. LJ (epsilon) untouched.
    Returns the NonbondedForce, or None if the System has none."""
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    if not nbfs:
        return None
    nbf = nbfs[0]
    # declare one global parameter per alchemical block
    for b in _alch_blocks(model):
        nbf.addGlobalParameter(lambda_name(b), 1.0)     

    # rewrite each alchemical atom's charge as λ_b * q_i
    for i in range(model.n_atoms):
        b = model.atom_block[i]
        if b == 0:
            continue
        # read original
        q, sig, eps = nbf.getParticleParameters(i)
        q0 = q.value_in_unit(_E)
        # base charge -> 0 (keep sigma/eps) 
        nbf.setParticleParameters(i, 0.0, sig, eps)     
        if q0 != 0.0:
            # +λ_b*q0 
            nbf.addParticleParameterOffset(lambda_name(b), i, q0, 0.0, 0.0)  

    # exclude same-site pairs
    add_same_site_exclusions(nbf, model)
    return nbf


# ---------------------------------------------------------------------
# Lennard-Jones
# ---------------------------------------------------------------------
# Each pair must be scaled by lambda_i * lambda_j (the product of the two atoms'
# lambdas). NonbondedForce cannot express this, since scaling epsilon yields 
# sqrt(lambda_i * lambda_j) instead.
#
# We therefore remove LJ for all alchemical pairs from NonbondedForce and move it
# into CustomNonbondedForce, where the energy expression is defined explicitly and
# multiplied by the desired lambda factor.
#
# One CustomNonbondedForce is created per pair "kind", each carrying a fixed lambda
# prefactor, with addInteractionGroup selecting the pairs it covers:
#     block b with environment, or block b with itself   ->  * lambda_b
#     block b with block c at a different site           ->  * lambda_b * lambda_c
#
# Atoms of the same site (competing substituents) are placed in no group, so they
# never interact. Plain environment-environment LJ remains in NonbondedForce.
# CustomNonbondedForce does expose dU/dlambda.
#
# Soft-core (model.softcore_alpha > 0, the default): pairs that can overlap -- block<->env and
# block<->block across sites -- use the Beutler form
#     U = L * 4*eps * [1/s^2 - 1/s],   s = (r/sigma)^6 + alpha*(1 - L)
# with L the pair's lambda product. At L = 1 this is plain LJ (lambda = 1 reproduces the base
# System); as L -> 0 it stays finite at r = 0, so a switched-off block that solvent has filled
# has a bounded dU/dlambda instead of the r^-12 wall (1e5-1e6 kJ/mol without it). Within-block
# (1-5+) pairs are softened too, with L = lambda_b, as in BLaDE: a switched-off block has its
# torsions and 1-4s off as well, so it can fold onto itself (MDM2 ILE: 1e4-1e5 kJ/mol spikes).
# Still disabled (see vault EXP-005): the long-range dispersion correction.


def _match_nonbonded_method(cnbf: mm.CustomNonbondedForce, nbf: mm.NonbondedForce):
    """Copy method / cutoff / switching from a NonbondedForce so the LJ matches."""
    NB, CN = mm.NonbondedForce, mm.CustomNonbondedForce
    m = nbf.getNonbondedMethod()
    if m == NB.NoCutoff:
        cnbf.setNonbondedMethod(CN.NoCutoff)
    elif m == NB.CutoffNonPeriodic:
        cnbf.setNonbondedMethod(CN.CutoffNonPeriodic)
        cnbf.setCutoffDistance(nbf.getCutoffDistance())
    else:  # CutoffPeriodic, Ewald, PME, LJPME -> real-space cutoff, periodic
        cnbf.setNonbondedMethod(CN.CutoffPeriodic)
        cnbf.setCutoffDistance(nbf.getCutoffDistance())
    if nbf.getUseSwitchingFunction():
        cnbf.setUseSwitchingFunction(True)
        cnbf.setSwitchingDistance(nbf.getSwitchingDistance())
    cnbf.setUseLongRangeCorrection(False)   # dispersion correction OFF (deferred)

# LJ energy expression, with Lorentz-Berthelot combining rules for sigma/epsilon
_LJ_EXPR = ("*4*epsilon*((sigma/r)^12-(sigma/r)^6);"
            "sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")


def _softcore_lj_expr(prefix: str, alpha: float) -> str:
    """Beutler soft-core LJ scaled by L = prefix. Applied to EVERY lambda-scaled pair, the
    within-block (1-5+) ones included, as BLaDE does (nbdirect.cu: rSoft = R*(1 - l_i) for a
    same-block pair): with lambda_b -> 0 the block's torsions and 1-4s are off too, so its own
    side chain can fold onto itself, and a plain within-block LJ then puts 1e4-1e5 kJ/mol spikes
    into dU/dlambda_b (MDM2 ILE)."""
    return (f"L*4*epsilon*(1/s^2-1/s);"
            f"s=(r/sigma)^6+({alpha})*(1-L);"
            f"L={prefix};"
            "sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")


def softcore_alpha(model) -> float:
    """Soft-core strength of the model (0 = plain linear scaling)."""
    return float(getattr(model, "softcore_alpha", 0.0) or 0.0)


def add_scaled_lj(system: mm.System, model) -> list[mm.CustomNonbondedForce]:
    """Move all alchemical-involving LJ from the NonbondedForce into grouped,
    block-scaled CustomNonbondedForce(s). env<->env LJ stays put. Returns the
    created forces."""
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    if not nbfs:
        return []
    nbf = nbfs[0]
    npart = system.getNumParticles()

    # capture original per-particle LJ (sigma, epsilon) BEFORE we zero anything
    lj = []
    for i in range(npart):
        _, sig, eps = nbf.getParticleParameters(i)
        lj.append((sig.value_in_unit(_NM), eps.value_in_unit(_KJ)))
    excl = sorted(_existing_exception_pairs(nbf))          # 1-2/1-3 + same-site + 1-4

    alch = _alch_blocks(model)
    env_atoms = [i for i in range(npart) if model.atom_block[i] == 0]
    block_atoms = {b: list(model.blocks[b].atoms) for b in alch}

    alpha = softcore_alpha(model)

    def _make(sig_tuple):
        if alpha > 0.0:
            f = mm.CustomNonbondedForce(_softcore_lj_expr(_lambda_prefix(sig_tuple), alpha))
        else:
            f = mm.CustomNonbondedForce(_lambda_prefix(sig_tuple) + _LJ_EXPR)
        f.setName("MSLDLJ_" + "_".join(map(str, sig_tuple)))
        _match_nonbonded_method(f, nbf)
        f.addPerParticleParameter("sigma")
        f.addPerParticleParameter("epsilon")
        for s, e in lj:
            f.addParticle([s, e])
        for a, c in excl:
            f.addExclusion(a, c)
        _register_lambdas(f, sig_tuple)         # globals + analytic dU/dlambda
        return f

    created = []
    # single-lambda: env<->block b and within block b
    for b in alch:
        f = _make((b,))
        if env_atoms and block_atoms[b]:
            f.addInteractionGroup(env_atoms, block_atoms[b])
        if len(block_atoms[b]) >= 2:
            f.addInteractionGroup(block_atoms[b], block_atoms[b])
        system.addForce(f)
        created.append(f)
    # product-lambda: block b <-> block c, different sites only
    for x in range(len(alch)):
        for y in range(x + 1, len(alch)):
            b, c = alch[x], alch[y]
            if model.blocks[b].site == model.blocks[c].site:
                continue                        # same site -> never interact
            f = _make((b, c))
            f.addInteractionGroup(block_atoms[b], block_atoms[c])
            system.addForce(f)
            created.append(f)

    # remove alchemical LJ from the stock NonbondedForce (keep charge & sigma)
    for i in range(npart):
        if model.atom_block[i] != 0:
            q, sig, eps = nbf.getParticleParameters(i)
            nbf.setParticleParameters(i, q, sig, 0.0)
    nbf.setUseDispersionCorrection(False)       # deferred
    return created


def _direct_space_screen(nbf: mm.NonbondedForce):
    """Real-space Coulomb kernel g(x) of the NonbondedForce (U_pair = KE*q1*q2*g(r)), as an
    OpenMM expression template in `{x}`, plus the CustomNonbondedForce method and cutoff."""
    NB, CN = mm.NonbondedForce, mm.CustomNonbondedForce
    method = nbf.getNonbondedMethod()
    if method == NB.NoCutoff:
        return "1/{x}", CN.NoCutoff, None
    rc = nbf.getCutoffDistance().value_in_unit(_NM)
    if method in (NB.PME, NB.Ewald, NB.LJPME):
        alpha = _ewald_alpha(nbf)
        return f"erfc({alpha}*{{x}})/{{x}}", CN.CutoffPeriodic, rc
    # reaction field (CutoffPeriodic / CutoffNonPeriodic); its constant shift cancels in a difference
    eps = nbf.getReactionFieldDielectric()
    krf = (eps - 1.0) / ((2.0 * eps + 1.0) * rc ** 3)
    cn = CN.CutoffPeriodic if method == NB.CutoffPeriodic else CN.CutoffNonPeriodic
    return f"(1/{{x}}+{krf}*{{x}}^2)", cn, rc


def add_softcore_electrostatics(system: mm.System, model, charges) -> list:
    """Soft-core the direct-space Coulomb of pairs that can overlap (block<->env, within a block
    (1-5+), and block<->block across sites), as a correction on top of the lambda-scaled
    NonbondedForce (+ within-block correction, which together give L*g(r) within a block):

        dU = L*KE*q1*q2*[g(r_sc) - g(r)],   r_sc = (r^6 + alpha*(1-L)*sigma_q^6)^(1/6)

    so that NonbondedForce + correction = soft-core real space + unchanged (smooth) reciprocal
    space. At L = 1 the correction is exactly 0. `charges` are the RAW (unscaled) charges. Must run
    after add_scaled_electrostatics (same-site exclusions) so the exclusion set matches the base
    force. Without it, LJ soft-core alone lets opposite charges collapse onto each other.
    """
    alpha = softcore_alpha(model)
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    if alpha <= 0.0 or not nbfs:
        return []
    nbf = nbfs[0]
    sig6 = float(getattr(model, "softcore_sigma_q", 0.3)) ** 6
    screen, cn_method, cutoff = _direct_space_screen(nbf)
    excl = sorted(_existing_exception_pairs(nbf))
    alch = _alch_blocks(model)
    env_atoms = [i for i in range(model.n_atoms) if model.atom_block[i] == 0]
    block_atoms = {b: list(model.blocks[b].atoms) for b in alch}
    charged = {b: any(charges[a] != 0.0 for a in block_atoms[b]) for b in alch}

    def _make(sig_tuple):
        prefix = _lambda_prefix(sig_tuple)
        f = mm.CustomNonbondedForce(
            f"L*{_KE}*q1*q2*({screen.format(x='rsc')}-{screen.format(x='r')});"
            f"rsc=(r^6+({alpha})*(1-L)*{sig6})^(1/6);"
            f"L={prefix}")
        f.setName("MSLDSCElec_" + "_".join(map(str, sig_tuple)))
        f.addPerParticleParameter("q")
        for q in charges:
            f.addParticle([q])
        for a, c in excl:
            f.addExclusion(a, c)
        f.setNonbondedMethod(cn_method)
        if cutoff is not None:
            f.setCutoffDistance(cutoff * _NM)
        f.setUseLongRangeCorrection(False)
        _register_lambdas(f, sig_tuple)
        return f

    env_charged = any(charges[a] != 0.0 for a in env_atoms)
    created = []
    for b in alch:
        if not charged[b]:
            continue
        with_env = bool(env_atoms) and env_charged
        within = len(block_atoms[b]) >= 2
        if not (with_env or within):
            continue
        f = _make((b,))
        if with_env:
            f.addInteractionGroup(env_atoms, block_atoms[b])
        if within:
            f.addInteractionGroup(block_atoms[b], block_atoms[b])
        system.addForce(f)
        created.append(f)
    for x in range(len(alch)):
        for y in range(x + 1, len(alch)):
            b, c = alch[x], alch[y]
            if model.blocks[b].site == model.blocks[c].site or not (charged[b] and charged[c]):
                continue
            f = _make((b, c))
            f.addInteractionGroup(block_atoms[b], block_atoms[c])
            system.addForce(f)
            created.append(f)
    return created


# ---------------------------------------------------------------------
# CHARMM tabulated LJ (NBFIX) -> block-scaled
# ---------------------------------------------------------------------
# When a CHARMM system has NBFIX, ParmEd/OpenMM put ALL Lennard-Jones into a single
# CustomNonbondedForce (energy uses per-type acoef/bcoef lookup tables), and ZERO the LJ
# in the NonbondedForce. So add_scaled_lj (which reads the NonbondedForce) finds nothing.
# We scale that tabulated force here, mirroring add_scaled_lj's grouping but reusing the
# CHARMM energy expression + copying the acoef/bcoef tables (so NBFIX is preserved):
#   base force  -> restricted to env<->env (interaction group)
#   env<->block b, within block b  -> lambda_b copy
#   block b<->c (different sites)   -> lambda_b*lambda_c copy
#   same-site block pairs           -> excluded (never grouped)
# Charges stay in the NonbondedForce (unaffected), so electrostatics is handled elsewhere.
# No-op on Amber (LJ lives in the NonbondedForce, not a CustomNonbondedForce).


def _copy_tabulated(fn):
    """Deep-copy an OpenMM tabulated function (the acoef/bcoef LJ tables)."""
    if isinstance(fn, mm.Discrete2DFunction):
        xs, ys, vals = fn.getFunctionParameters()
        return mm.Discrete2DFunction(xs, ys, vals)
    if isinstance(fn, mm.Discrete1DFunction):
        (vals,) = fn.getFunctionParameters()
        return mm.Discrete1DFunction(vals)
    raise NotImplementedError(f"cannot copy tabulated function of type {type(fn).__name__}")


def _is_parmed_ab_table(per_params, tabs) -> bool:
    """True for ParmEd's tabulated LJ: U = (a/r^6)^2 - b/r^6 with a = acoef(type1, type2),
    b = bcoef(type1, type2) (a^2 = 4 eps sigma^12, b = 4 eps sigma^6, NBFIX already folded in)."""
    return per_params == ["type"] and {n for n, _ in tabs} == {"acoef", "bcoef"}


def _softcore_ab_expr(prefix: str, alpha: float) -> str:
    """Beutler soft-core in ParmEd's acoef/bcoef form, the tabulated analogue of
    _softcore_lj_expr: 4 eps = b^2/a^2 and sigma^6 = a^2/b per PAIR, so NBFIX pairs soften with
    their own sigma. Within-block pairs are softened too, as in _softcore_lj_expr. A pair with b = 0 (zero-LJ types) keeps the plain form, which is then 0 or pure repulsion."""
    return (f"L*select(bb, (bb*bb/a2)*(1/(s*s)-1/s), a2/(r6*r6)-bb/r6);"
            f"s=r6*bb/a2+({alpha})*(1-L);"
            f"a2=aa*aa+1e-30; r6=r^6;"
            f"aa=acoef(type1, type2); bb=bcoef(type1, type2);"
            f"L={prefix}")


def add_scaled_charmm_lj(system: mm.System, model) -> list[mm.CustomNonbondedForce]:
    """Block-scale a CHARMM-style tabulated LJ CustomNonbondedForce. No-op on Amber."""
    created = []
    alch_set = {a for a in range(model.n_atoms) if model.atom_block[a] != 0}
    if not alch_set:
        return created
    alch = _alch_blocks(model)
    env_atoms = [i for i in range(model.n_atoms) if model.atom_block[i] == 0]
    block_atoms = {b: list(model.blocks[b].atoms) for b in alch}
    npart = system.getNumParticles()

    # non-MSLD CustomNonbondedForce spanning all particles = the system LJ force
    originals = [system.getForce(i) for i in range(system.getNumForces())
                 if isinstance(system.getForce(i), mm.CustomNonbondedForce)
                 and not (system.getForce(i).getName() or "").startswith("MSLD")
                 and system.getForce(i).getNumParticles() == npart]

    alpha = softcore_alpha(model)
    # Every nonbonded force must carry the same exclusion set on CPU/OpenCL/CUDA: the
    # NonbondedForce now also excludes the same-site pairs, so add those here too.
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    nb_excl = _existing_exception_pairs(nbfs[0]) if nbfs else set()
    for f in originals:
        expr = f.getEnergyFunction()
        per_params = [f.getPerParticleParameterName(k)
                      for k in range(f.getNumPerParticleParameters())]
        gparams = [(f.getGlobalParameterName(g), f.getGlobalParameterDefaultValue(g))
                   for g in range(f.getNumGlobalParameters())]
        pvals = [list(f.getParticleParameters(i)) for i in range(npart)]
        tabs = [(f.getTabulatedFunctionName(t), f.getTabulatedFunction(t))
                for t in range(f.getNumTabulatedFunctions())]
        own = {tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions())}
        for a, c in sorted(nb_excl - own):
            f.addExclusion(a, c)
        excls = sorted(own | nb_excl)
        method = f.getNonbondedMethod()
        cutoff = f.getCutoffDistance()
        use_switch = f.getUseSwitchingFunction()
        switch = f.getSwitchingDistance()
        main, sep, tail = expr.partition(";")
        softcore = alpha > 0.0 and _is_parmed_ab_table(per_params, tabs)
        if alpha > 0.0 and not softcore:
            import warnings
            warnings.warn("soft-core needs the ParmEd acoef/bcoef LJ table; this tabulated LJ "
                          "uses plain linear lambda scaling", stacklevel=2)

        def _make(sig_tuple, name):
            if softcore:
                g = mm.CustomNonbondedForce(_softcore_ab_expr(_lambda_prefix(sig_tuple), alpha))
            else:
                g = mm.CustomNonbondedForce(f"({_lambda_prefix(sig_tuple)})*({main}){sep}{tail}")
            g.setName(name)
            for pn in per_params:
                g.addPerParticleParameter(pn)
            for gn, dv in gparams:
                g.addGlobalParameter(gn, dv)
            for tname, tfn in tabs:
                g.addTabulatedFunction(tname, _copy_tabulated(tfn))
            for pv in pvals:
                g.addParticle(pv)
            for a, c in excls:
                g.addExclusion(a, c)
            g.setNonbondedMethod(method)
            if method != mm.CustomNonbondedForce.NoCutoff:
                g.setCutoffDistance(cutoff)
            if use_switch:
                g.setUseSwitchingFunction(True)
                g.setSwitchingDistance(switch)
            g.setUseLongRangeCorrection(False)      # unsupported with interaction groups
            _register_lambdas(g, sig_tuple)
            return g

        # single-lambda: env<->block b and within block b (both soft-cored, as in add_scaled_lj)
        for b in alch:
            g = _make((b,), "MSLDCharmmLJ_%d" % b)
            if env_atoms and block_atoms[b]:
                g.addInteractionGroup(env_atoms, block_atoms[b])
            if len(block_atoms[b]) >= 2:
                g.addInteractionGroup(block_atoms[b], block_atoms[b])
            system.addForce(g)
            created.append(g)
        # product-lambda: block b <-> block c, different sites only
        for x in range(len(alch)):
            for y in range(x + 1, len(alch)):
                b, c = alch[x], alch[y]
                if model.blocks[b].site == model.blocks[c].site:
                    continue
                g = _make((b, c), "MSLDCharmmLJ_%d_%d" % (b, c))
                g.addInteractionGroup(block_atoms[b], block_atoms[c])
                system.addForce(g)
                created.append(g)

        # restrict the base force to env<->env only (it keeps the NBFIX tables)
        f.addInteractionGroup(env_atoms, env_atoms)
        f.setUseLongRangeCorrection(False)
    return created


# ---------------------------------------------------------------------
# 1-4 exceptions
# ---------------------------------------------------------------------
# An exception in NonbondedForce replaces the normal interaction for one specific
# atom pair. Two kinds exist:
#   full exclusion : charge and epsilon set to 0 -> pair does not interact.
#                    Covers 1-2 and 1-3 neighbours, plus same-site pairs.
#   scaled 1-4     : charge and epsilon carry the force field's reduced "1-4"
#                    values, applied to atoms three bonds apart.
#
# Only the scaled 1-4 interactions need lambda scaling. When both atoms fall under
# a single lambda (same block, or one atom in the environment), the dependence is
# again linear in lambda_b, so an exception parameter offset handles it exactly:
#     charge  = lambda_b * (1-4 charge)
#     epsilon = lambda_b * (1-4 epsilon)
# (dU/dlambda here is finite difference, as elsewhere in NonbondedForce.)
#
# A 1-4 whose atoms lie in different sites (requiring lambda_i * lambda_j) or in
# the same site is rare or disallowed, so it currently raises an error.

def add_scaled_14_exceptions(system: mm.System, model) -> int:
    """Scale alchemical 1-4 exceptions by their single block lambda via exception
    parameter offsets. Returns the number scaled. Raises on cross-site (product)
    or same-site 1-4 exceptions."""
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    if not nbfs:
        return 0
    nbf = nbfs[0]
    scaled = 0
    for k in range(nbf.getNumExceptions()):
        p1, p2, cp, sig, eps = nbf.getExceptionParameters(k)
        cpv = cp.value_in_unit(_E**2)
        epsv = eps.value_in_unit(_KJ)
        if cpv == 0.0 and epsv == 0.0:
            continue                                 # full exclusion, nothing to scale
        if model.atom_block[p1] == 0 and model.atom_block[p2] == 0:
            continue                                 # environment 1-4, leave as-is
        cls = model.classify_pair(p1, p2)
        if cls[0] == "single":
            b = cls[1]
            nbf.setExceptionParameters(k, p1, p2, 0.0, sig, 0.0)   # base -> 0
            nbf.addExceptionParameterOffset(lambda_name(b), k, cpv, 0.0, epsv)
            scaled += 1
        elif cls[0] == "product":
            raise NotImplementedError(
                f"cross-site 1-4 exception scaling is deferred (rare); atoms {p1},{p2}")
        elif cls[0] == "exclude":
            raise NotImplementedError(
                f"same-site 1-4 exception (illegal MSLD topology?); atoms {p1},{p2}")
    return scaled


# ---------------------------------------------------------------------
# Biasing potentials (ALF)
# ---------------------------------------------------------------------
# MSLD needs biases on lambda to flatten the landscape (ALF). These are pure
# functions of the lambda globals -- no atom positions -- so we add ONE
# one-particle CustomCompoundBondForce whose energy is the whole bias sum written in
# terms of lambda{b}. Its atom force is zero (no position in the expression), and unlike
# CustomExternalForce it supports addEnergyParameterDerivative, so dU/dlambda of the
# biases is analytic. That keeps the finite-difference remainder purely quadratic in
# lambda (the NonbondedForce), which is what lets the integrator use a large FD step.
#
# Fixed bias : sum_b  bias_b * lambda_b            (linear)
# Variable   : ALF coupling biases between two blocks i, j (BLaDE types):
#     6  quadratic : k * li * lj
#     8  endpoint  : k * li * lj / (li + l0)
#     10 skew      : k * lj * (1 - exp(l0 * li))




def apply_stock_nonbonded(system: mm.System, model) -> dict:
    """Scale the System's nonbonded terms with stock forces (the OpenMM-MSLD stack). Puts the NonbondedForce
    in FD_FORCE_GROUP. Must run after the bonded scaling and before the biases, like the plugin path."""
    info = {}
    info["within_block_elec_corrections"] = [f.getName() for f in add_within_block_elec_correction(system, model)]
    nbfs = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    raw_q = ([nbfs[0].getParticleParameters(i)[0].value_in_unit(_E) for i in range(system.getNumParticles())]
             if nbfs else [])
    nbf = add_scaled_electrostatics(system, model)
    info["scaled_lj_forces"] = [f.getName() for f in add_scaled_lj(system, model)]
    info["softcore_elec_forces"] = [f.getName() for f in add_softcore_electrostatics(system, model, raw_q)]
    info["scaled_charmm_lj_forces"] = [f.getName() for f in add_scaled_charmm_lj(system, model)]
    info["scaled_14_exceptions"] = add_scaled_14_exceptions(system, model)
    if nbf is not None:
        for f in system.getForces():
            if f.getForceGroup() == FD_FORCE_GROUP and f is not nbf:
                raise RuntimeError(f"force group {FD_FORCE_GROUP} is reserved for the NonbondedForce")
        nbf.setForceGroup(FD_FORCE_GROUP)
        info["fd_force_group"] = FD_FORCE_GROUP
    return info
