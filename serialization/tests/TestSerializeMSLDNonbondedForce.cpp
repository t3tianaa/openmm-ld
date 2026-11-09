/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "MSLDNonbondedForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerMSLDSerializationProxies();

void testSerialization() {
    MSLDNonbondedForce force;
    force.setForceGroup(3);
    force.setNonbondedMethod(MSLDNonbondedForce::PME);
    force.setCutoffDistance(0.9);
    force.setUseSwitchingFunction(true);
    force.setSwitchingDistance(0.8);
    force.setReactionFieldDielectric(50.0);
    force.setEwaldErrorTolerance(1e-4);
    force.setPMEParameters(3.1, 20, 22, 24);
    force.setExceptionsUsePeriodicBoundaryConditions(true);
    force.setSoftcoreAlpha(0.3);
    force.setSoftcoreSigmaQ(0.25);
    force.addBlock(1, 0.4);
    force.addBlock(1, 0.6);
    force.addBlock(2, 1.0);
    force.addParticle(0.1, 0.3, 0.5, 0);
    force.addParticle(-0.2, 0.31, 0.6, 1);
    force.addParticle(0.3, 0.32, 0.7, 2);
    force.addParticle(-0.4, 0.33, 0.8, 3);
    force.setParticleLJType(1, 7);
    force.setParticleLJType(2, 8);
    force.addNBFixPair(7, 8, 0.29, 0.45);
    force.addException(0, 1, 0.01, 0.3, 0.2);
    force.addException(1, 3, 0.0, 1.0, 0.0);

    stringstream buffer;
    XmlSerializer::serialize<MSLDNonbondedForce>(&force, "Force", buffer);
    MSLDNonbondedForce* copy = XmlSerializer::deserialize<MSLDNonbondedForce>(buffer);

    MSLDNonbondedForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getUseSwitchingFunction(), force2.getUseSwitchingFunction());
    ASSERT_EQUAL(force.getSwitchingDistance(), force2.getSwitchingDistance());
    ASSERT_EQUAL(force.getReactionFieldDielectric(), force2.getReactionFieldDielectric());
    ASSERT_EQUAL(force.getEwaldErrorTolerance(), force2.getEwaldErrorTolerance());
    ASSERT_EQUAL(force.getExceptionsUsePeriodicBoundaryConditions(), force2.getExceptionsUsePeriodicBoundaryConditions());
    ASSERT_EQUAL(force.getSoftcoreAlpha(), force2.getSoftcoreAlpha());
    ASSERT_EQUAL(force.getSoftcoreSigmaQ(), force2.getSoftcoreSigmaQ());
    double a1, a2;
    int x1, y1, z1, x2, y2, z2;
    force.getPMEParameters(a1, x1, y1, z1);
    force2.getPMEParameters(a2, x2, y2, z2);
    ASSERT_EQUAL(a1, a2);
    ASSERT_EQUAL(x1, x2);
    ASSERT_EQUAL(y1, y2);
    ASSERT_EQUAL(z1, z2);
    ASSERT_EQUAL(force.getNumBlocks(), force2.getNumBlocks());
    for (int b = 1; b < force.getNumBlocks(); b++) {
        ASSERT_EQUAL(force.getBlockSite(b), force2.getBlockSite(b));
        ASSERT_EQUAL(force.getBlockDefaultLambda(b), force2.getBlockDefaultLambda(b));
    }
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double q1, s1, e1, q2, s2, e2;
        int b1, b2;
        force.getParticleParameters(i, q1, s1, e1, b1);
        force2.getParticleParameters(i, q2, s2, e2, b2);
        ASSERT_EQUAL(q1, q2);
        ASSERT_EQUAL(s1, s2);
        ASSERT_EQUAL(e1, e2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(force.getParticleLJType(i), force2.getParticleLJType(i));
    }
    ASSERT_EQUAL(force.getNumNBFixPairs(), force2.getNumNBFixPairs());
    for (int k = 0; k < force.getNumNBFixPairs(); k++) {
        int t1a, t2a, t1b, t2b;
        double sa, ea, sb, eb;
        force.getNBFixPairParameters(k, t1a, t2a, sa, ea);
        force2.getNBFixPairParameters(k, t1b, t2b, sb, eb);
        ASSERT_EQUAL(t1a, t1b);
        ASSERT_EQUAL(t2a, t2b);
        ASSERT_EQUAL(sa, sb);
        ASSERT_EQUAL(ea, eb);
    }
    ASSERT_EQUAL(force.getNumExceptions(), force2.getNumExceptions());
    for (int k = 0; k < force.getNumExceptions(); k++) {
        int i1, j1, i2, j2;
        double q1, s1, e1, q2, s2, e2;
        force.getExceptionParameters(k, i1, j1, q1, s1, e1);
        force2.getExceptionParameters(k, i2, j2, q2, s2, e2);
        ASSERT_EQUAL(i1, i2);
        ASSERT_EQUAL(j1, j2);
        ASSERT_EQUAL(q1, q2);
        ASSERT_EQUAL(s1, s2);
        ASSERT_EQUAL(e1, e2);
    }
    delete copy;
}

int main() {
    try {
        registerMSLDSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
