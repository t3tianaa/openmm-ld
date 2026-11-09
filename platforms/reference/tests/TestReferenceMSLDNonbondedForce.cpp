/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 *                                                                            *
 * Tests of MSLDNonbondedForce on the Reference (and CPU) platform:           *
 *   - a 4-atom system against a hand calculation                             *
 *   - lambda = 1 against OpenMM::NonbondedForce (NoCutoff, RF, PME)          *
 *   - finite-difference checks of forces and dU/dlambda for every method     *
 *   - neighbour-list path against the brute-force pair loop                  *
 * -------------------------------------------------------------------------- */

#include "MSLDNonbondedForce.h"
#include "openmm/Context.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerMSLDReferenceKernelFactories();

// OpenMM's exact 1/(4 pi eps0); the spec rounds it to 138.935456 (relative difference ~1e-8).
static const double KE = ONE_4PI_EPS0;

#ifdef __APPLE__
static const char* CPU_PLUGIN_NAME = "libOpenMMCPU.dylib";
#elif defined(_WIN32)
static const char* CPU_PLUGIN_NAME = "OpenMMCPU.dll";
#else
static const char* CPU_PLUGIN_NAME = "libOpenMMCPU.so";
#endif

static string platformName = "Reference";

static double getEnergy(Context& context) {
    return context.getState(State::Energy).getPotentialEnergy();
}

static map<string, double> getDerivs(Context& context) {
    return context.getState(State::ParameterDerivatives).getEnergyParameterDerivatives();
}

// ---------------------------------------------------------------------------
// Hand calculation
// ---------------------------------------------------------------------------

static double softLJ(double r, double sig, double eps, double L, double alpha) {
    double s = pow(r/sig, 6.0) + alpha*(1.0-L);
    return L*4.0*eps*(1.0/(s*s) - 1.0/s);
}

static double softCoul(double r, double qq, double L, double alpha, double sigmaQ) {
    double rsc = pow(pow(r, 6.0) + alpha*(1.0-L)*pow(sigmaQ, 6.0), 1.0/6.0);
    return L*KE*qq/rsc;
}

static double dist(const Vec3& a, const Vec3& b) {
    Vec3 d = a-b;
    return sqrt(d.dot(d));
}

/**
 * Atoms: 0 environment, 1 block 1 (site 1), 2 block 2 (site 2), 3 block 3 (site 1).
 * (0,1) is a 1-4 exception; (1,3) is excluded automatically (same site, different block).
 */
static double handEnergy(const vector<Vec3>& x, const double* q, const double* sg, const double* ep, double l1, double l2, double l3,
        double alpha, double sigmaQ, double qq14, double sig14, double eps14) {
    double U = 0.0;
    // (0,1): 1-4, L = lambda1, no soft-core
    double r = dist(x[0], x[1]);
    U += l1*(KE*qq14/r + 4*eps14*(pow(sig14/r, 12) - pow(sig14/r, 6)));
    // (0,2): env + block 2, L = lambda2
    r = dist(x[0], x[2]);
    U += softLJ(r, 0.5*(sg[0]+sg[2]), sqrt(ep[0]*ep[2]), l2, alpha) + softCoul(r, q[0]*q[2], l2, alpha, sigmaQ);
    // (0,3): env + block 3, L = lambda3
    r = dist(x[0], x[3]);
    U += softLJ(r, 0.5*(sg[0]+sg[3]), sqrt(ep[0]*ep[3]), l3, alpha) + softCoul(r, q[0]*q[3], l3, alpha, sigmaQ);
    // (1,2): sites 1 and 2, L = lambda1*lambda2
    r = dist(x[1], x[2]);
    U += softLJ(r, 0.5*(sg[1]+sg[2]), sqrt(ep[1]*ep[2]), l1*l2, alpha) + softCoul(r, q[1]*q[2], l1*l2, alpha, sigmaQ);
    // (1,3): same site -> nothing
    // (2,3): sites 2 and 1, L = lambda2*lambda3
    r = dist(x[2], x[3]);
    U += softLJ(r, 0.5*(sg[2]+sg[3]), sqrt(ep[2]*ep[3]), l2*l3, alpha) + softCoul(r, q[2]*q[3], l2*l3, alpha, sigmaQ);
    return U;
}

void testHandCalculation() {
    const double q[] = {0.4, -0.3, 0.25, -0.5};
    const double sg[] = {0.30, 0.32, 0.28, 0.35};
    const double ep[] = {0.5, 0.7, 0.3, 0.6};
    const double qq14 = 0.07, sig14 = 0.29, eps14 = 0.2;
    const double alpha = 0.5, sigmaQ = 0.3;
    const double l1 = 0.3, l2 = 0.6, l3 = 0.7;
    vector<Vec3> x(4);
    x[0] = Vec3(0, 0, 0);
    x[1] = Vec3(0.31, 0.05, 0);
    x[2] = Vec3(0.1, 0.38, 0.07);
    x[3] = Vec3(-0.2, 0.15, 0.33);

    System system;
    MSLDNonbondedForce* force = new MSLDNonbondedForce();
    ASSERT_EQUAL(1, force->addBlock(1));
    ASSERT_EQUAL(2, force->addBlock(2));
    ASSERT_EQUAL(3, force->addBlock(1));
    const int blocks[] = {0, 1, 2, 3};
    for (int i = 0; i < 4; i++) {
        system.addParticle(1.0);
        force->addParticle(q[i], sg[i], ep[i], blocks[i]);
    }
    force->addException(0, 1, qq14, sig14, eps14);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, Platform::getPlatformByName(platformName));
    context.setPositions(x);
    context.setParameter("lambda1", l1);
    context.setParameter("lambda2", l2);
    context.setParameter("lambda3", l3);

    State state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);
    double expected = handEnergy(x, q, sg, ep, l1, l2, l3, alpha, sigmaQ, qq14, sig14, eps14);
    ASSERT_EQUAL_TOL(expected, state.getPotentialEnergy(), 1e-10);

    // dU/dlambda against a central difference of the hand formula
    const double h = 1e-5;
    map<string, double> derivs = state.getEnergyParameterDerivatives();
    double d1 = (handEnergy(x, q, sg, ep, l1+h, l2, l3, alpha, sigmaQ, qq14, sig14, eps14) - handEnergy(x, q, sg, ep, l1-h, l2, l3, alpha, sigmaQ, qq14, sig14, eps14))/(2*h);
    double d2 = (handEnergy(x, q, sg, ep, l1, l2+h, l3, alpha, sigmaQ, qq14, sig14, eps14) - handEnergy(x, q, sg, ep, l1, l2-h, l3, alpha, sigmaQ, qq14, sig14, eps14))/(2*h);
    double d3 = (handEnergy(x, q, sg, ep, l1, l2, l3+h, alpha, sigmaQ, qq14, sig14, eps14) - handEnergy(x, q, sg, ep, l1, l2, l3-h, alpha, sigmaQ, qq14, sig14, eps14))/(2*h);
    ASSERT_EQUAL_TOL(d1, derivs["lambda1"], 1e-7);
    ASSERT_EQUAL_TOL(d2, derivs["lambda2"], 1e-7);
    ASSERT_EQUAL_TOL(d3, derivs["lambda3"], 1e-7);

    // forces against a central difference of the hand formula
    const vector<Vec3>& forces = state.getForces();
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 3; k++) {
            vector<Vec3> xp = x, xm = x;
            xp[i][k] += h;
            xm[i][k] -= h;
            double fd = -(handEnergy(xp, q, sg, ep, l1, l2, l3, alpha, sigmaQ, qq14, sig14, eps14) - handEnergy(xm, q, sg, ep, l1, l2, l3, alpha, sigmaQ, qq14, sig14, eps14))/(2*h);
            ASSERT_EQUAL_TOL(fd, forces[i][k], 1e-6);
        }

    // derivatives are not accumulated across evaluations
    map<string, double> again = getDerivs(context);
    ASSERT_EQUAL_TOL(derivs["lambda1"], again["lambda1"], 1e-12);

    // at lambda = 1 the soft-core shift vanishes: plain LJ + Coulomb
    context.setParameter("lambda1", 1.0);
    context.setParameter("lambda2", 1.0);
    context.setParameter("lambda3", 1.0);
    ASSERT_EQUAL_TOL(handEnergy(x, q, sg, ep, 1, 1, 1, 0.0, sigmaQ, qq14, sig14, eps14), getEnergy(context), 1e-10);

    // soft-core off: plain linear scaling
    force->setSoftcoreAlpha(0.0);
    context.reinitialize(true);
    context.setParameter("lambda1", l1);
    context.setParameter("lambda2", l2);
    context.setParameter("lambda3", l3);
    ASSERT_EQUAL_TOL(handEnergy(x, q, sg, ep, l1, l2, l3, 0.0, sigmaQ, qq14, sig14, eps14), getEnergy(context), 1e-10);
}

// ---------------------------------------------------------------------------
// Random test system: environment + 2 sites (blocks 1,2 at site 1; block 3 at site 2)
// ---------------------------------------------------------------------------

struct TestSystem {
    System system;
    MSLDNonbondedForce* msld;
    vector<Vec3> positions;
};

static void buildSystem(TestSystem& ts, MSLDNonbondedForce::NonbondedMethod method, bool useSwitch, double boxSize, bool nbfix) {
    const int numParticles = 30;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    ts.system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    ts.msld = new MSLDNonbondedForce();
    ts.msld->addBlock(1);
    ts.msld->addBlock(1);
    ts.msld->addBlock(2);
    ts.msld->setNonbondedMethod(method);
    ts.msld->setCutoffDistance(1.0);
    ts.msld->setUseSwitchingFunction(useSwitch);
    ts.msld->setSwitchingDistance(0.8);
    ts.msld->setReactionFieldDielectric(78.3);
    ts.msld->setEwaldErrorTolerance(1e-5);
    for (int i = 0; i < numParticles; i++) {
        ts.system.addParticle(1.0);
        int block = (i < 12 ? 0 : 1 + (i-12)/6);    // 12 env, then 6 per block (block 1..3)
        double q = 0.8*(genrand_real2(sfmt)-0.5);
        double sig = 0.25 + 0.1*genrand_real2(sfmt);
        double eps = 0.2 + 0.5*genrand_real2(sfmt);
        ts.msld->addParticle(q, sig, eps, block);
        if (nbfix)
            ts.msld->setParticleLJType(i, i%3);
        // spread atoms out on a jittered grid so no pair is closer than ~0.2 nm
        int gx = i%4, gy = (i/4)%4, gz = i/16;
        ts.positions.push_back(Vec3(0.3*gx + 0.08*genrand_real2(sfmt), 0.3*gy + 0.08*genrand_real2(sfmt), 0.3*gz + 0.08*genrand_real2(sfmt)));
    }
    if (nbfix) {
        ts.msld->addNBFixPair(0, 1, 0.27, 0.9);
        ts.msld->addNBFixPair(2, 2, 0.33, 0.15);
    }
    // exclusions and 1-4s: env-env, env-block, within block
    ts.msld->addException(0, 1, 0.0, 1.0, 0.0);
    ts.msld->addException(2, 3, 0.03, 0.3, 0.2);
    ts.msld->addException(4, 12, 0.0, 1.0, 0.0);
    ts.msld->addException(5, 13, -0.02, 0.28, 0.3);
    ts.msld->addException(14, 15, 0.0, 1.0, 0.0);
    ts.msld->addException(16, 17, 0.04, 0.31, 0.25);
    ts.msld->addException(24, 25, 0.05, 0.3, 0.2);
    ts.msld->addException(6, 26, 0.01, 0.3, 0.1);
    ts.system.addForce(ts.msld);
}

static void setLambdas(Context& context, double l1, double l2, double l3) {
    context.setParameter("lambda1", l1);
    context.setParameter("lambda2", l2);
    context.setParameter("lambda3", l3);
}

/**
 * At lambda = 1 the MSLD force is a NonbondedForce with the same-site cross-block pairs excluded.
 */
void testMatchesNonbondedForce(MSLDNonbondedForce::NonbondedMethod method, bool useSwitch) {
    TestSystem ts;
    buildSystem(ts, method, useSwitch, 3.0, false);
    MSLDNonbondedForce& msld = *ts.msld;

    System refSystem;
    refSystem.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    NonbondedForce* nb = new NonbondedForce();
    nb->setNonbondedMethod((NonbondedForce::NonbondedMethod) method);
    nb->setCutoffDistance(msld.getCutoffDistance());
    nb->setUseSwitchingFunction(useSwitch);
    nb->setSwitchingDistance(msld.getSwitchingDistance());
    nb->setReactionFieldDielectric(msld.getReactionFieldDielectric());
    nb->setEwaldErrorTolerance(msld.getEwaldErrorTolerance());
    nb->setUseDispersionCorrection(false);
    for (int i = 0; i < msld.getNumParticles(); i++) {
        double q, s, e;
        int b;
        msld.getParticleParameters(i, q, s, e, b);
        refSystem.addParticle(1.0);
        nb->addParticle(q, s, e);
    }
    for (int k = 0; k < msld.getNumExceptions(); k++) {
        int i, j;
        double q, s, e;
        msld.getExceptionParameters(k, i, j, q, s, e);
        nb->addException(i, j, q, s, e);
    }
    for (int i = 0; i < msld.getNumParticles(); i++)
        for (int j = i+1; j < msld.getNumParticles(); j++) {
            double q, s, e;
            int bi, bj;
            msld.getParticleParameters(i, q, s, e, bi);
            msld.getParticleParameters(j, q, s, e, bj);
            if (bi != 0 && bj != 0 && bi != bj && msld.getBlockSite(bi) == msld.getBlockSite(bj))
                nb->addException(i, j, 0.0, 1.0, 0.0);
        }
    refSystem.addForce(nb);

    VerletIntegrator i1(0.001), i2(0.001);
    Context c1(ts.system, i1, Platform::getPlatformByName(platformName));
    Context c2(refSystem, i2, Platform::getPlatformByName("Reference"));
    c1.setPositions(ts.positions);
    c2.setPositions(ts.positions);
    State s1 = c1.getState(State::Energy | State::Forces);
    State s2 = c2.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), 1e-8);
    for (int i = 0; i < msld.getNumParticles(); i++)
        ASSERT_EQUAL_VEC(s2.getForces()[i], s1.getForces()[i], 1e-7);
    if (method == MSLDNonbondedForce::PME) {
        double a1, a2;
        int x1, y1, z1, x2, y2, z2;
        msld.getPMEParametersInContext(c1, a1, x1, y1, z1);
        nb->getPMEParametersInContext(c2, a2, x2, y2, z2);
        ASSERT_EQUAL(a2, a1);
        ASSERT_EQUAL(x2, x1);
        ASSERT_EQUAL(y2, y1);
        ASSERT_EQUAL(z2, z1);
    }
}

/**
 * Forces and dU/dlambda against central finite differences of the energy.
 */
void testFiniteDifferences(MSLDNonbondedForce::NonbondedMethod method, bool useSwitch, bool nbfix) {
    TestSystem ts;
    buildSystem(ts, method, useSwitch, 3.0, nbfix);
    VerletIntegrator integrator(0.001);
    Context context(ts.system, integrator, Platform::getPlatformByName(platformName));
    context.setPositions(ts.positions);
    const double l[] = {0.0, 0.35, 0.55, 0.8};
    setLambdas(context, l[1], l[2], l[3]);
    State state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);
    map<string, double> derivs = state.getEnergyParameterDerivatives();

    // dU/dlambda_b
    const double hl = 1e-4;
    for (int b = 1; b <= 3; b++) {
        double lp[4] = {0, l[1], l[2], l[3]}, lm[4] = {0, l[1], l[2], l[3]};
        lp[b] += hl;
        lm[b] -= hl;
        setLambdas(context, lp[1], lp[2], lp[3]);
        double ep = getEnergy(context);
        setLambdas(context, lm[1], lm[2], lm[3]);
        double em = getEnergy(context);
        double fd = (ep-em)/(2*hl);
        ASSERT_EQUAL_TOL(fd, derivs[MSLDNonbondedForce::getLambdaName(b)], 1e-6);
    }
    setLambdas(context, l[1], l[2], l[3]);

    // forces: displace all atoms along the force direction
    const vector<Vec3>& forces = state.getForces();
    double norm = 0.0;
    for (int i = 0; i < (int) forces.size(); i++)
        norm += forces[i].dot(forces[i]);
    norm = sqrt(norm);
    const double step = 1e-4;
    vector<Vec3> xp(ts.positions), xm(ts.positions);
    for (int i = 0; i < (int) forces.size(); i++) {
        Vec3 dx = forces[i]*(step/norm);
        xp[i] += dx;
        xm[i] -= dx;
    }
    context.setPositions(xp);
    double ep = getEnergy(context);
    context.setPositions(xm);
    double em = getEnergy(context);
    ASSERT_EQUAL_TOL(norm, (em-ep)/(2*step), 1e-5);
}

/**
 * A 1-4 exception between two different blocks must be rejected.
 */
void testIllegalException() {
    System system;
    MSLDNonbondedForce* force = new MSLDNonbondedForce();
    force->addBlock(1);
    force->addBlock(2);
    system.addParticle(1.0);
    system.addParticle(1.0);
    force->addParticle(0.1, 0.3, 0.5, 1);
    force->addParticle(0.1, 0.3, 0.5, 2);
    force->addException(0, 1, 0.01, 0.3, 0.5);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    bool threw = false;
    try {
        Context context(system, integrator, Platform::getPlatformByName(platformName));
    }
    catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);
}

/**
 * updateParametersInContext() changes charges and LJ parameters.
 */
void testUpdateParameters() {
    TestSystem ts;
    buildSystem(ts, MSLDNonbondedForce::NoCutoff, false, 3.0, false);
    VerletIntegrator integrator(0.001);
    Context context(ts.system, integrator, Platform::getPlatformByName(platformName));
    context.setPositions(ts.positions);
    setLambdas(context, 0.4, 0.6, 0.5);
    double e1 = getEnergy(context);
    double q, s, e;
    int b;
    ts.msld->getParticleParameters(13, q, s, e, b);
    ts.msld->setParticleParameters(13, q+0.3, s, e, b);
    ts.msld->updateParametersInContext(context);
    double e2 = getEnergy(context);
    ASSERT(fabs(e2-e1) > 1e-3);

    VerletIntegrator integrator2(0.001);
    Context fresh(ts.system, integrator2, Platform::getPlatformByName(platformName));
    fresh.setPositions(ts.positions);
    setLambdas(fresh, 0.4, 0.6, 0.5);
    ASSERT_EQUAL_TOL(getEnergy(fresh), e2, 1e-10);
}

/**
 * The neighbour-list path against the O(N^2) pair loop: 512 atoms on a jittered grid, some wrapped
 * outside the box, in a rectangular or triclinic periodic box or without periodicity.
 */
void testNeighborListMatchesBruteForce(MSLDNonbondedForce::NonbondedMethod method, bool triclinic) {
    const int perSide = 8, numParticles = perSide*perSide*perSide;
    const double spacing = 0.4, box = perSide*spacing;
    const Vec3 a(box, 0, 0), b(triclinic ? 0.8 : 0.0, box, 0), c(triclinic ? 0.8 : 0.0, triclinic ? 0.8 : 0.0, box);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(1, sfmt);
    System system;
    system.setDefaultPeriodicBoxVectors(a, b, c);
    MSLDNonbondedForce* force = new MSLDNonbondedForce();
    force->addBlock(1);
    force->addBlock(1);
    force->addBlock(2);
    force->setNonbondedMethod(method);
    force->setCutoffDistance(1.0);
    force->setUseSwitchingFunction(true);
    force->setSwitchingDistance(0.85);
    force->setEwaldErrorTolerance(5e-4);
    vector<Vec3> positions;
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        int block = (i%8 == 0 ? 1 + (i/8)%3 : 0);
        force->addParticle(0.8*(genrand_real2(sfmt)-0.5), 0.25+0.1*genrand_real2(sfmt), 0.2+0.5*genrand_real2(sfmt), block);
        Vec3 x = a*((i%perSide+0.25*genrand_real2(sfmt))/perSide)
               + b*(((i/perSide)%perSide+0.25*genrand_real2(sfmt))/perSide)
               + c*((i/(perSide*perSide)+0.25*genrand_real2(sfmt))/perSide);
        if (i%5 == 0 && method != MSLDNonbondedForce::CutoffNonPeriodic)
            x += a*(i%2 == 0 ? 1.0 : -1.0) + c;
        positions.push_back(x);
    }
    for (int i = 0; i+1 < numParticles; i += 16)
        force->addException(i, i+1, (i%32 == 0 ? 0.0 : 0.03), 0.3, (i%32 == 0 ? 0.0 : 0.2));
    system.addForce(force);

    State states[2];
    for (int k = 0; k < 2; k++) {
        if (k == 1)
            setenv("MSLD_REFERENCE_BRUTE_FORCE_PAIRS", "1", 1);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, Platform::getPlatformByName(platformName));
        unsetenv("MSLD_REFERENCE_BRUTE_FORCE_PAIRS");
        context.setPositions(positions);
        setLambdas(context, 0.35, 0.65, 0.8);
        states[k] = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);
    }
    ASSERT_EQUAL_TOL(states[1].getPotentialEnergy(), states[0].getPotentialEnergy(), 1e-12);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(states[1].getForces()[i], states[0].getForces()[i], 1e-12);
    map<string, double> d0 = states[0].getEnergyParameterDerivatives(), d1 = states[1].getEnergyParameterDerivatives();
    for (int b = 1; b <= 3; b++)
        ASSERT_EQUAL_TOL(d1[MSLDNonbondedForce::getLambdaName(b)], d0[MSLDNonbondedForce::getLambdaName(b)], 1e-12);
}

void runAll() {
    testHandCalculation();
    testIllegalException();
    testUpdateParameters();
    testMatchesNonbondedForce(MSLDNonbondedForce::NoCutoff, false);
    testMatchesNonbondedForce(MSLDNonbondedForce::CutoffNonPeriodic, true);
    testMatchesNonbondedForce(MSLDNonbondedForce::CutoffPeriodic, false);
    testMatchesNonbondedForce(MSLDNonbondedForce::CutoffPeriodic, true);
    testMatchesNonbondedForce(MSLDNonbondedForce::PME, false);
    testMatchesNonbondedForce(MSLDNonbondedForce::PME, true);
    testFiniteDifferences(MSLDNonbondedForce::NoCutoff, false, false);
    testFiniteDifferences(MSLDNonbondedForce::NoCutoff, false, true);
    testFiniteDifferences(MSLDNonbondedForce::CutoffNonPeriodic, true, false);
    testFiniteDifferences(MSLDNonbondedForce::CutoffPeriodic, true, true);
    testFiniteDifferences(MSLDNonbondedForce::PME, false, false);
    testFiniteDifferences(MSLDNonbondedForce::PME, true, true);
    testNeighborListMatchesBruteForce(MSLDNonbondedForce::CutoffNonPeriodic, false);
    testNeighborListMatchesBruteForce(MSLDNonbondedForce::CutoffPeriodic, false);
    testNeighborListMatchesBruteForce(MSLDNonbondedForce::CutoffPeriodic, true);
    testNeighborListMatchesBruteForce(MSLDNonbondedForce::PME, false);
    testNeighborListMatchesBruteForce(MSLDNonbondedForce::PME, true);
}

int main() {
    try {
        // Load OpenMM's CPU platform if it is there, then register this plugin's kernels.
        try {
            Platform::loadPluginLibrary(string(OPENMM_PLUGIN_DIR) + "/" + CPU_PLUGIN_NAME);
        }
        catch (const exception& e) {
            cout << "CPU platform not loaded: " << e.what() << endl;
        }
        registerMSLDReferenceKernelFactories();
        cout << "Reference platform" << endl;
        runAll();
        bool haveCpu = false;
        for (int i = 0; i < Platform::getNumPlatforms(); i++)
            if (Platform::getPlatform(i).getName() == "CPU")
                haveCpu = true;
        if (haveCpu) {
            cout << "CPU platform" << endl;
            platformName = "CPU";
            runAll();
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
