"""Block-scaled forces: bonded terms per lambda signature, biases, restraints, and the nonbonded force.

Ported from OpenMM-MSLD msld/forces.py and msld/plugin.py. Bonded, bias and restraint forces are stock
OpenMM custom forces whose energy carries lambda<b>. All nonbonded terms go through one
MSLDNonbondedForce. Block b's coupling is the Context parameter lambda<b>; the environment (block 0) has none.
"""
from __future__ import annotations

import os
from collections import defaultdict

import openmm as mm
from openmm import unit

_NM = unit.nanometer
_RAD = unit.radian
_KJ = unit.kilojoule_per_mole
_KJ_PER_NM2 = _KJ / _NM**2
_KJ_PER_RAD2 = _KJ / _RAD**2
_E = unit.elementary_charge

PLUGIN_FORCE_NAME = "MSLDNonbondedForce"


# --------------------------------------------------------------------- 
# Helpers 
# --------------------------------------------------------------------- 
def lambda_name(b: int) -> str:
    """Context global-parameter name for block b's coupling."""
    # block 1 -> "lambda1"
    return f"lambda{b}"


def _forces_of_type(system: mm.System, cls):
    """Return all forces of a given type in a System."""
    return [system.getForce(i) for i in range(system.getNumForces())
            if isinstance(system.getForce(i), cls)]


def _lambda_prefix(signature: tuple[int, ...]) -> str:
    """'lambda1' or 'lambda1*lambda3' etc. for the energy expression."""
    return "*".join(lambda_name(b) for b in signature)


def _register_lambdas(force, signature: tuple[int, ...]) -> None:
    """Declare block's global parameter and request its energy derivative."""
    for b in signature:
        force.addGlobalParameter(lambda_name(b), 1.0)       # declare the global (default 1.0)
        force.addEnergyParameterDerivative(lambda_name(b))  # dU/dlambda_b


# --------------------------------------------------------------------- 
# Bonds 
# --------------------------------------------------------------------- 
def add_scaled_bonds(system: mm.System, model) -> list[mm.CustomBondForce]:
    """HarmonicBondForce -> block-scaled CustomBondForce(s)."""
    created = []
    for hbf in _forces_of_type(system, mm.HarmonicBondForce):
        groups = defaultdict(list)
        # Classify each bond, and remove the scaled ones from the base force
        for i in range(hbf.getNumBonds()):
            p1, p2, length, k = hbf.getBondParameters(i)    # read bond i
            sig = tuple(model.classify_bonded([p1, p2]))    # which λ's scale it?
            if not sig:
                continue    # env-only -> keep in base force
            groups[sig].append((p1, p2, length.value_in_unit(_NM),
                                k.value_in_unit(_KJ_PER_NM2)))
            hbf.setBondParameters(i, p1, p2, length, 0.0 * k)   # zero it in the base force

        # build one CustomBondForce per signature
        for sig, bonds in groups.items():
            f = mm.CustomBondForce(f"{_lambda_prefix(sig)}*0.5*k*(r-length)^2")
            f.setName("MSLDBond_" + "_".join(map(str, sig)))
            _register_lambdas(f, sig)
            f.addPerBondParameter("length")
            f.addPerBondParameter("k")
            for p1, p2, length, k in bonds:
                f.addBond(p1, p2, [length, k])
            system.addForce(f)
            created.append(f)
    return created


# --------------------------------------------------------------------- 
# Angles 
# --------------------------------------------------------------------- 
def add_scaled_angles(system: mm.System, model) -> list[mm.CustomAngleForce]:
    """HarmonicAngleForce -> block-scaled CustomAngleForce(s)."""
    created = []
    for haf in _forces_of_type(system, mm.HarmonicAngleForce):
        groups = defaultdict(list)
        for i in range(haf.getNumAngles()):
            a1, a2, a3, angle, k = haf.getAngleParameters(i)
            sig = tuple(model.classify_bonded([a1, a2, a3]))
            if not sig:
                continue
            groups[sig].append((a1, a2, a3, angle.value_in_unit(_RAD),
                                k.value_in_unit(_KJ_PER_RAD2)))
            haf.setAngleParameters(i, a1, a2, a3, angle, 0.0 * k)
        for sig, angles in groups.items():
            f = mm.CustomAngleForce(f"{_lambda_prefix(sig)}*0.5*k*(theta-theta0)^2")
            f.setName("MSLDAngle_" + "_".join(map(str, sig)))
            _register_lambdas(f, sig)
            f.addPerAngleParameter("theta0")
            f.addPerAngleParameter("k")
            for a1, a2, a3, theta0, k in angles:
                f.addAngle(a1, a2, a3, [theta0, k])
            system.addForce(f)
            created.append(f)
    return created


# --------------------------------------------------------------------- 
# Torsions 
# --------------------------------------------------------------------- 
def add_scaled_torsions(system: mm.System, model) -> list[mm.CustomTorsionForce]:
    """PeriodicTorsionForce -> block-scaled CustomTorsionForce(s)."""
    created = []
    for ptf in _forces_of_type(system, mm.PeriodicTorsionForce):
        groups = defaultdict(list)
        for i in range(ptf.getNumTorsions()):
            a1, a2, a3, a4, per, phase, k = ptf.getTorsionParameters(i)
            sig = tuple(model.classify_bonded([a1, a2, a3, a4]))
            if not sig:
                continue
            groups[sig].append((a1, a2, a3, a4, int(per),
                                phase.value_in_unit(_RAD), k.value_in_unit(_KJ)))
            ptf.setTorsionParameters(i, a1, a2, a3, a4, per, phase, 0.0 * k)
        for sig, tors in groups.items():
            f = mm.CustomTorsionForce(
                f"{_lambda_prefix(sig)}*k*(1+cos(per*theta-phase))")
            f.setName("MSLDTorsion_" + "_".join(map(str, sig)))
            _register_lambdas(f, sig)
            f.addPerTorsionParameter("per")
            f.addPerTorsionParameter("phase")
            f.addPerTorsionParameter("k")
            for a1, a2, a3, a4, per, phase, k in tors:
                f.addTorsion(a1, a2, a3, a4, [per, phase, k])
            system.addForce(f)
            created.append(f)
    return created


# ---------------------------------------------------------------------
# Impropers (CHARMM harmonic impropers -> CustomTorsionForce)
# ---------------------------------------------------------------------
# CHARMM force fields have harmonic improper dihedrals. ParmEd/OpenMM represent them in
# a CustomTorsionForce, not the PeriodicTorsionForce that add_scaled_torsions handles. 
# Urey-Bradley terms are a separate HarmonicBondForce and are already scaled by add_scaled_bonds.
#
# We cannot cleanly zero one torsion inside a generic CustomTorsionForce (we do not know
# which parameter is the force constant), so instead we rebuild: read the force's energy
# expression + parameters, drop the original force, and re-add its torsions grouped by
# lambda signature - env terms unscaled, alchemical terms in a CustomTorsionForce whose
# energy is lambda_prefix*(original). These custom forces expose analytic dU/dlambda.


def add_scaled_impropers(system: mm.System, model) -> list[mm.CustomTorsionForce]:
    """Block-scale alchemical CustomTorsionForce terms (CHARMM harmonic impropers).

    Only forces that actually touch alchemical atoms are rebuilt, others are left as-is.
    Returns the created (scaled + unscaled-env) forces.
    """
    created = []
    # capture (index, force). OpenMM returns a fresh wrapper each call, so we must
    # track the index - `getForce(i) is f` never matches, even for the same force.
    # Skip MSLD-named CustomTorsionForces: add_scaled_torsions turns PeriodicTorsionForce
    # into CustomTorsionForce (MSLDTorsion_*), and those are already lambda-scaled - we
    # must not scale them again.
    originals = [(i, system.getForce(i)) for i in range(system.getNumForces())
                 if isinstance(system.getForce(i), mm.CustomTorsionForce)
                 and not (system.getForce(i).getName() or "").startswith("MSLD")]
    remove_indices = []
    for idx, f in originals:
        # classify every torsion; skip this force entirely if none are alchemical
        groups = defaultdict(list)
        has_alch = False
        for i in range(f.getNumTorsions()):
            p = f.getTorsionParameters(i)
            a1, a2, a3, a4 = p[0], p[1], p[2], p[3]
            params = list(p[4])
            sig = tuple(model.classify_bonded([a1, a2, a3, a4]))
            groups[sig].append((a1, a2, a3, a4, params))
            if sig:
                has_alch = True
        if not has_alch:
            continue

        expr = f.getEnergyFunction()
        per_params = [f.getPerTorsionParameterName(k)
                      for k in range(f.getNumPerTorsionParameters())]
        gparams = [(f.getGlobalParameterName(g), f.getGlobalParameterDefaultValue(g))
                   for g in range(f.getNumGlobalParameters())]
        pbc = f.usesPeriodicBoundaryConditions()
        main, sep, tail = expr.partition(";")     # scale only the value, not the a=b defs

        def _make(prefix: str, name: str):
            e = f"({prefix})*({main}){sep}{tail}" if prefix else expr
            g = mm.CustomTorsionForce(e)
            g.setName(name)
            for pn in per_params:
                g.addPerTorsionParameter(pn)
            for gn, dv in gparams:
                g.addGlobalParameter(gn, dv)
            g.setUsesPeriodicBoundaryConditions(pbc)
            return g

        base_name = f.getName() or "Improper"
        if groups.get(()):                         # env impropers -> unscaled copy
            g = _make("", base_name + "_env")
            for a1, a2, a3, a4, params in groups[()]:
                g.addTorsion(a1, a2, a3, a4, params)
            system.addForce(g)
            created.append(g)
        for sig, tors in groups.items():           # alchemical -> lambda-scaled
            if not sig:
                continue
            g = _make(_lambda_prefix(sig), "MSLDImproper_" + "_".join(map(str, sig)))
            _register_lambdas(g, sig)
            for a1, a2, a3, a4, params in tors:
                g.addTorsion(a1, a2, a3, a4, params)
            system.addForce(g)
            created.append(g)
        remove_indices.append(idx)

    # drop the originals by index. New forces were appended at the end, so the original
    # indices are still valid; remove them high-to-low so earlier indices don't shift.
    for idx in sorted(remove_indices, reverse=True):
        system.removeForce(idx)
    return created


# ---------------------------------------------------------------------
# Biasing potentials (ALF)
# ---------------------------------------------------------------------
# MSLD needs biases on lambda to flatten the landscape (ALF). These are pure
# functions of the lambda globals -- no atom positions -- so we add ONE
# one-particle CustomCompoundBondForce whose energy is the whole bias sum written in
# terms of lambda{b}. Its atom force is zero (no position in the expression), and unlike
# CustomExternalForce it supports addEnergyParameterDerivative, so dU/dlambda of the
# biases is analytic.
#
# Fixed bias : sum_b  bias_b * lambda_b            (linear)
# Variable   : ALF coupling biases between two blocks i, j (BLaDE types):
#     6  quadratic : k * li * lj
#     8  endpoint  : k * li * lj / (li + l0)
#     10 skew      : k * lj * (1 - exp(l0 * li))


def bias_coef_name(b: int) -> str:
    """Context global holding block b's fixed linear bias coefficient (tunable by ALF)."""
    return f"biascoef{b}"


def variable_bias_k_name(m: int) -> str:
    """Context global holding the m-th variable bias' k coefficient (tunable by ALF)."""
    return f"biask{m}"


def _variable_bias_term(vb, kname) -> str:
    li, lj = lambda_name(vb.i), lambda_name(vb.j)
    l0 = vb.l0
    if vb.type == 6:
        return f"{kname}*{li}*{lj}"
    if vb.type == 8:
        return f"{kname}*{li}*{lj}/({li}+({l0}))"
    if vb.type == 10:
        return f"{kname}*{lj}*(1-exp(({l0})*{li}))"
    raise NotImplementedError(
        f"variable bias type {vb.type} not implemented (have 6, 8, 10)")


def add_biases(system: mm.System, model) -> mm.CustomCompoundBondForce | None:
    """Add fixed + variable lambda biases as one position-independent force.

    Every bias COEFFICIENT is a Context global parameter (`biascoef{b}` for the fixed linear bias,
    `biask{m}` for the m-th variable bias), initialized to the model's current value -- so ALF can
    retune them live with `builder.set_biases` (no System/Context rebuild). The λ globals appear in
    the expression with addEnergyParameterDerivative, so the bias gradient is analytic.

    Returns the force, or None if there is nothing to add. With `model.bias_flattening` set, a
    tunable fixed-bias term is created for EVERY alchemical block even at coefficient 0, so ALF has
    a knob to turn from the start; otherwise only non-zero fixed biases (and any variable biases)
    create terms, so a plain run adds no bias force at all (unchanged default behavior).
    """
    flatten = getattr(model, "bias_flattening", False)
    terms, coefs, lam_refs = [], [], set()
    for b in range(1, model.n_blocks):
        bias = model.blocks[b].lambda_bias
        if bias != 0.0 or flatten:
            name = bias_coef_name(b)
            terms.append(f"{name}*{lambda_name(b)}")
            coefs.append((name, bias))
            lam_refs.add(b)
    for m, vb in enumerate(model.variable_biases):
        kname = variable_bias_k_name(m)
        terms.append(_variable_bias_term(vb, kname))
        coefs.append((kname, vb.k))
        lam_refs.update((vb.i, vb.j))
    if not terms:
        return None

    f = mm.CustomCompoundBondForce(1, " + ".join(terms))
    f.setName("MSLDBias")
    for name, val in coefs:
        f.addGlobalParameter(name, val)          # tunable coefficient globals
    for b in sorted(lam_refs):
        f.addGlobalParameter(lambda_name(b), 1.0)
        f.addEnergyParameterDerivative(lambda_name(b))   # analytic dU/dlambda of the bias
    f.addBond([0], [])            # dummy anchor; energy does not depend on its position
    system.addForce(f)
    return f


# ---------------------------------------------------------------------
# Atom restraints (CATS) 
# ---------------------------------------------------------------------
# Keep each substituent's atoms near their own centroid so a "switched-off"
# substituent (lambda ~ 0) does not drift away. BLaDE: U = 0.5*k*sum_i |x_i - xbar|^2.
# That centroid restraint is IDENTICAL to a harmonic term over every in-group
# pair with force constant k/N, so we just add pair bonds -- no centroid needed:
#     0.5*k*sum_i |x_i - xbar|^2  ==  (k/(2N)) * sum_{i<j} |x_i - x_j|^2
# The restraint is NOT lambda-scaled (always on), matching BLaDE.


def add_atom_restraints(system: mm.System, model) -> mm.CustomBondForce | None:
    """Add centroid restraints for each group in model.atom_restraints."""
    if not model.atom_restraints:
        return None
    f = mm.CustomBondForce("0.5*kr*r^2")
    f.setName("MSLDAtomRestraint")
    f.addPerBondParameter("kr")
    added = 0
    for group in model.atom_restraints:
        n = len(group)
        if n < 2:
            continue
        kr = model.k_restraint / n               # per-pair constant (see identity above)
        for a in range(n):
            for c in range(a + 1, n):
                f.addBond(group[a], group[c], [kr])
                added += 1
    if added == 0:
        return None
    system.addForce(f)
    return f


# ---------------------------------------------------------------------
# Nonbonded: one MSLDNonbondedForce
# ---------------------------------------------------------------------
def _is_parmed_ab_table(per_params, tabs) -> bool:
    """True for ParmEd's tabulated LJ: U = (a/r^6)^2 - b/r^6 with a = acoef(type1, type2),
    b = bcoef(type1, type2) (a^2 = 4 eps sigma^12, b = 4 eps sigma^6, NBFIX already folded in)."""
    return per_params == ["type"] and {n for n, _ in tabs} == {"acoef", "bcoef"}


_loaded_library = None


def load_msldplugin():
    """Import msldplugin and, when MSLD_REFERENCE_PLUGIN is set, load its kernel library once."""
    global _loaded_library
    try:
        import msldplugin
    except ImportError as e:
        raise ImportError("openmm_ld needs the msldplugin module (cmake --install, or "
                          "PYTHONPATH=<build>/python)") from e
    library = os.environ.get("MSLD_REFERENCE_PLUGIN")
    if library and library != _loaded_library:
        msldplugin.loadReferencePlugin(library)
        _loaded_library = library
    return msldplugin


def is_plugin_force(force) -> bool:
    """True for the plugin's force. System.getForce returns it as a plain Force, so go by name."""
    return force.getName() == PLUGIN_FORCE_NAME


def _is_tabulated_lj(f, n_particles) -> bool:
    """A non-MSLD CustomNonbondedForce spanning all particles: the CHARMM tabulated LJ."""
    return (isinstance(f, mm.CustomNonbondedForce) and f.getNumParticles() == n_particles
            and not (f.getName() or "").startswith("MSLD"))


def _apply_tabulated_lj(force, lj):
    """LJ types and per-type-pair (sigma, epsilon) from ParmEd's acoef/bcoef tables, in which
    U = (a/r^6)^2 - b/r^6, so sigma^6 = a^2/b and 4 eps = b^2/a^2. Every pair of types goes in as an
    override, so the table is reproduced exactly (NBFIX already folded in)."""
    per_params = [lj.getPerParticleParameterName(k) for k in range(lj.getNumPerParticleParameters())]
    tabs = {lj.getTabulatedFunctionName(t): lj.getTabulatedFunction(t)
            for t in range(lj.getNumTabulatedFunctions())}
    if not _is_parmed_ab_table(per_params, list(tabs.items())):
        raise NotImplementedError("MSLDNonbondedForce needs ParmEd's acoef/bcoef LJ table; this tabulated "
                                  f"LJ has parameters {per_params} and tables {sorted(tabs)}")
    nx, ny, acoef = tabs["acoef"].getFunctionParameters()
    _, _, bcoef = tabs["bcoef"].getFunctionParameters()
    if nx != ny:
        raise NotImplementedError(f"MSLDNonbondedForce: non-square LJ table ({nx} x {ny})")

    for t1 in range(nx):
        for t2 in range(t1, nx):
            a, b = acoef[t1 + nx * t2], bcoef[t1 + nx * t2]
            if a != acoef[t2 + nx * t1] or b != bcoef[t2 + nx * t1]:
                raise NotImplementedError(f"MSLDNonbondedForce: asymmetric LJ table at types {t1}, {t2}")
            if b > 0.0 and a != 0.0:
                force.addNBFixPair(t1, t2, (a * a / b) ** (1.0 / 6.0), b * b / (4.0 * a * a))
            elif a == 0.0 and b == 0.0:
                force.addNBFixPair(t1, t2, 1.0, 0.0)
            else:
                raise NotImplementedError(f"MSLDNonbondedForce: LJ table entry a={a}, b={b} at types {t1}, "
                                          f"{t2} has no (sigma, epsilon) form")
    for i in range(lj.getNumParticles()):
        force.setParticleLJType(i, int(lj.getParticleParameters(i)[0]))


def replace_nonbonded_with_plugin(system: mm.System, model) -> tuple:
    """Replace the system's nonbonded forces by one MSLDNonbondedForce, in place.

    Returns (the force, the number of alchemical 1-4 exceptions it scales). Raises where the stock
    route does (cross-site or same-site 1-4 exceptions) and on anything the plugin cannot represent.
    """
    msldplugin = load_msldplugin()
    NB, P = mm.NonbondedForce, msldplugin.MSLDNonbondedForce
    nbfs = _forces_of_type(system, mm.NonbondedForce)
    if len(nbfs) != 1:
        raise ValueError(f"MSLDNonbondedForce needs exactly one NonbondedForce, found {len(nbfs)}")
    nbf = nbfs[0]
    if nbf.getNumParticleParameterOffsets() or nbf.getNumExceptionParameterOffsets():
        raise ValueError("MSLDNonbondedForce: the NonbondedForce already has parameter offsets")
    methods = {NB.NoCutoff: P.NoCutoff, NB.CutoffNonPeriodic: P.CutoffNonPeriodic,
               NB.CutoffPeriodic: P.CutoffPeriodic, NB.PME: P.PME}
    if nbf.getNonbondedMethod() not in methods:
        raise NotImplementedError(f"MSLDNonbondedForce: nonbonded method {nbf.getNonbondedMethod()} "
                                  "is not supported (Ewald, LJPME)")
    n_particles = system.getNumParticles()
    tabulated = [f for f in system.getForces() if _is_tabulated_lj(f, n_particles)]
    if len(tabulated) > 1:
        raise NotImplementedError("MSLDNonbondedForce: more than one tabulated LJ force")
    lj_force = tabulated[0] if tabulated else None

    force = P()
    force.setName(PLUGIN_FORCE_NAME)
    force.setNonbondedMethod(methods[nbf.getNonbondedMethod()])
    force.setCutoffDistance(nbf.getCutoffDistance().value_in_unit(_NM))
    # the switching function acts on LJ only, so it comes from whichever force holds the LJ
    switch_source = lj_force if lj_force is not None else nbf
    force.setUseSwitchingFunction(switch_source.getUseSwitchingFunction())
    force.setSwitchingDistance(switch_source.getSwitchingDistance().value_in_unit(_NM))
    force.setReactionFieldDielectric(nbf.getReactionFieldDielectric())
    force.setEwaldErrorTolerance(nbf.getEwaldErrorTolerance())
    alpha, nx, ny, nz = nbf.getPMEParameters()
    force.setPMEParameters(alpha.value_in_unit(_NM ** -1), nx, ny, nz)
    force.setExceptionsUsePeriodicBoundaryConditions(nbf.getExceptionsUsePeriodicBoundaryConditions())
    force.setSoftcoreAlpha(float(getattr(model, "softcore_alpha", 0.0) or 0.0))
    force.setSoftcoreSigmaQ(float(model.softcore_sigma_q))
    if lj_force is not None and nbf.getNonbondedMethod() != NB.NoCutoff:
        if abs(lj_force.getCutoffDistance().value_in_unit(_NM)
               - nbf.getCutoffDistance().value_in_unit(_NM)) > 1e-12:
            raise NotImplementedError("MSLDNonbondedForce: tabulated LJ and NonbondedForce cutoffs differ")

    for b in range(1, model.n_blocks):
        if force.addBlock(model.blocks[b].site) != b:
            raise RuntimeError("MSLDNonbondedForce: block numbering mismatch")
    for i in range(n_particles):
        q, sig, eps = nbf.getParticleParameters(i)
        force.addParticle(q.value_in_unit(_E), sig.value_in_unit(_NM),
                          eps.value_in_unit(_KJ), model.atom_block[i])

    scaled = 0
    excluded = set()
    for k in range(nbf.getNumExceptions()):
        p1, p2, cp, sig, eps = nbf.getExceptionParameters(k)
        cpv, epsv = cp.value_in_unit(_E ** 2), eps.value_in_unit(_KJ)
        if cpv != 0.0 or epsv != 0.0:
            cls = model.classify_pair(p1, p2)
            if cls[0] == "product":
                raise NotImplementedError(f"cross-site 1-4 exception; atoms {p1},{p2}")
            if cls[0] == "exclude":
                raise NotImplementedError(f"same-site 1-4 exception (illegal MSLD topology?); "
                                          f"atoms {p1},{p2}")
            scaled += cls[0] == "single"
        force.addException(p1, p2, cpv, sig.value_in_unit(_NM), epsv)
        excluded.add((min(p1, p2), max(p1, p2)))

    if lj_force is not None:
        _apply_tabulated_lj(force, lj_force)
        for k in range(lj_force.getNumExclusions()):
            p1, p2 = lj_force.getExclusionParticles(k)
            if (min(p1, p2), max(p1, p2)) not in excluded:
                raise NotImplementedError(f"MSLDNonbondedForce: atoms {p1},{p2} are excluded from the "
                                          "tabulated LJ but not from the NonbondedForce")

    for i in reversed(range(system.getNumForces())):
        f = system.getForce(i)
        if isinstance(f, mm.NonbondedForce) or _is_tabulated_lj(f, n_particles):
            system.removeForce(i)
    system.addForce(force)
    return force, scaled


def exception_pairs(system: mm.System) -> set:
    """All (i, j), i < j, with an exception on the plugin force."""
    msldplugin = load_msldplugin()
    pairs = set()
    for f in system.getForces():
        if is_plugin_force(f):
            p = msldplugin.MSLDNonbondedForce.cast(f)
            for k in range(p.getNumExceptions()):
                i, j, *_ = p.getExceptionParameters(k)
                pairs.add((min(i, j), max(i, j)))
    return pairs
