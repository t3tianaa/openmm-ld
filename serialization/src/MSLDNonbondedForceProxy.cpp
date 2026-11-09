/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "MSLDNonbondedForceProxy.h"
#include "MSLDNonbondedForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

MSLDNonbondedForceProxy::MSLDNonbondedForceProxy() : SerializationProxy("MSLDNonbondedForce") {
}

void MSLDNonbondedForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const MSLDNonbondedForce& force = *reinterpret_cast<const MSLDNonbondedForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setBoolProperty("useSwitchingFunction", force.getUseSwitchingFunction());
    node.setDoubleProperty("switchingDistance", force.getSwitchingDistance());
    node.setDoubleProperty("rfDielectric", force.getReactionFieldDielectric());
    node.setDoubleProperty("ewaldTolerance", force.getEwaldErrorTolerance());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("alpha", alpha);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    node.setBoolProperty("exceptionsUsePeriodic", force.getExceptionsUsePeriodicBoundaryConditions());
    node.setDoubleProperty("softcoreAlpha", force.getSoftcoreAlpha());
    node.setDoubleProperty("softcoreSigmaQ", force.getSoftcoreSigmaQ());
    SerializationNode& blocks = node.createChildNode("Blocks");
    for (int b = 1; b < force.getNumBlocks(); b++)
        blocks.createChildNode("Block").setIntProperty("site", force.getBlockSite(b)).setDoubleProperty("lambda", force.getBlockDefaultLambda(b));
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        int block;
        force.getParticleParameters(i, charge, sigma, epsilon, block);
        particles.createChildNode("Particle").setDoubleProperty("q", charge).setDoubleProperty("sig", sigma).setDoubleProperty("eps", epsilon)
                .setIntProperty("block", block).setIntProperty("type", force.getParticleLJType(i));
    }
    SerializationNode& nbfix = node.createChildNode("NBFix");
    for (int k = 0; k < force.getNumNBFixPairs(); k++) {
        int t1, t2;
        double sigma, epsilon;
        force.getNBFixPairParameters(k, t1, t2, sigma, epsilon);
        nbfix.createChildNode("Pair").setIntProperty("t1", t1).setIntProperty("t2", t2).setDoubleProperty("sig", sigma).setDoubleProperty("eps", epsilon);
    }
    SerializationNode& exceptions = node.createChildNode("Exceptions");
    for (int k = 0; k < force.getNumExceptions(); k++) {
        int p1, p2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(k, p1, p2, chargeProd, sigma, epsilon);
        exceptions.createChildNode("Exception").setIntProperty("p1", p1).setIntProperty("p2", p2).setDoubleProperty("q", chargeProd)
                .setDoubleProperty("sig", sigma).setDoubleProperty("eps", epsilon);
    }
}

void* MSLDNonbondedForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != 1)
        throw OpenMMException("Unsupported version number");
    MSLDNonbondedForce* force = new MSLDNonbondedForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((MSLDNonbondedForce::NonbondedMethod) node.getIntProperty("method"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setUseSwitchingFunction(node.getBoolProperty("useSwitchingFunction"));
        force->setSwitchingDistance(node.getDoubleProperty("switchingDistance"));
        force->setReactionFieldDielectric(node.getDoubleProperty("rfDielectric"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldTolerance"));
        force->setPMEParameters(node.getDoubleProperty("alpha"), node.getIntProperty("nx"), node.getIntProperty("ny"), node.getIntProperty("nz"));
        force->setExceptionsUsePeriodicBoundaryConditions(node.getBoolProperty("exceptionsUsePeriodic"));
        force->setSoftcoreAlpha(node.getDoubleProperty("softcoreAlpha"));
        force->setSoftcoreSigmaQ(node.getDoubleProperty("softcoreSigmaQ"));
        const vector<SerializationNode>& blocks = node.getChildNode("Blocks").getChildren();
        for (int b = 0; b < (int) blocks.size(); b++)
            force->addBlock(blocks[b].getIntProperty("site"), blocks[b].getDoubleProperty("lambda"));
        const vector<SerializationNode>& particles = node.getChildNode("Particles").getChildren();
        for (int i = 0; i < (int) particles.size(); i++) {
            const SerializationNode& p = particles[i];
            force->addParticle(p.getDoubleProperty("q"), p.getDoubleProperty("sig"), p.getDoubleProperty("eps"), p.getIntProperty("block"));
            force->setParticleLJType(i, p.getIntProperty("type"));
        }
        const vector<SerializationNode>& nbfix = node.getChildNode("NBFix").getChildren();
        for (int k = 0; k < (int) nbfix.size(); k++) {
            const SerializationNode& f = nbfix[k];
            force->addNBFixPair(f.getIntProperty("t1"), f.getIntProperty("t2"), f.getDoubleProperty("sig"), f.getDoubleProperty("eps"));
        }
        const vector<SerializationNode>& exceptions = node.getChildNode("Exceptions").getChildren();
        for (int k = 0; k < (int) exceptions.size(); k++) {
            const SerializationNode& e = exceptions[k];
            force->addException(e.getIntProperty("p1"), e.getIntProperty("p2"), e.getDoubleProperty("q"), e.getDoubleProperty("sig"), e.getDoubleProperty("eps"));
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
