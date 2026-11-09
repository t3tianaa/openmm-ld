/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "internal/MSLDNonbondedForceImpl.h"
#include "MSLDKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <sstream>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

MSLDNonbondedForceImpl::MSLDNonbondedForceImpl(const MSLDNonbondedForce& owner) : owner(owner) {
}

MSLDNonbondedForceImpl::~MSLDNonbondedForceImpl() {
}

void MSLDNonbondedForceImpl::validate(const System& system, const MSLDNonbondedForce& force) {
    if (force.getNumParticles() != system.getNumParticles())
        throw OpenMMException("MSLDNonbondedForce must have exactly as many particles as the System it belongs to.");
    int numBlocks = force.getNumBlocks();
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        int block;
        force.getParticleParameters(i, charge, sigma, epsilon, block);
        if (block < 0 || block >= numBlocks) {
            stringstream msg;
            msg << "MSLDNonbondedForce: particle " << i << " has an illegal block index " << block;
            throw OpenMMException(msg.str());
        }
    }
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int p1, p2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, p1, p2, chargeProd, sigma, epsilon);
        if (p1 < 0 || p1 >= force.getNumParticles() || p2 < 0 || p2 >= force.getNumParticles() || p1 == p2) {
            stringstream msg;
            msg << "MSLDNonbondedForce: illegal particle index for exception " << i;
            throw OpenMMException(msg.str());
        }
        if (chargeProd == 0.0 && epsilon == 0.0)
            continue;
        double q, s, e;
        int b1, b2;
        force.getParticleParameters(p1, q, s, e, b1);
        force.getParticleParameters(p2, q, s, e, b2);
        if (b1 != 0 && b2 != 0 && b1 != b2) {
            stringstream msg;
            msg << "MSLDNonbondedForce: exception " << i << " (particles " << p1 << ", " << p2 << ") is a 1-4 interaction between blocks "
                << b1 << " and " << b2 << "; 1-4 terms between different blocks are not supported";
            throw OpenMMException(msg.str());
        }
    }
    MSLDNonbondedForce::NonbondedMethod method = force.getNonbondedMethod();
    if (method != MSLDNonbondedForce::NoCutoff) {
        double cutoff = force.getCutoffDistance();
        if (cutoff <= 0)
            throw OpenMMException("MSLDNonbondedForce: the cutoff distance must be positive");
        if (force.getUseSwitchingFunction() && (force.getSwitchingDistance() < 0 || force.getSwitchingDistance() >= cutoff))
            throw OpenMMException("MSLDNonbondedForce: the switching distance must satisfy 0 <= r_switch < r_cutoff");
    }
    if (force.usesPeriodicBoundaryConditions()) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = force.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("MSLDNonbondedForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
}

void MSLDNonbondedForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    validate(system, owner);
    kernel = context.getPlatform().createKernel(CalcMSLDNonbondedForceKernel::Name(), context);
    kernel.getAs<CalcMSLDNonbondedForceKernel>().initialize(system, owner);
}

double MSLDNonbondedForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcMSLDNonbondedForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

map<string, double> MSLDNonbondedForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int b = 1; b < owner.getNumBlocks(); b++)
        parameters[MSLDNonbondedForce::getLambdaName(b)] = owner.getBlockDefaultLambda(b);
    return parameters;
}

vector<string> MSLDNonbondedForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcMSLDNonbondedForceKernel::Name());
    return names;
}

void MSLDNonbondedForceImpl::updateParametersInContext(ContextImpl& context) {
    validate(context.getSystem(), owner);
    kernel.getAs<CalcMSLDNonbondedForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void MSLDNonbondedForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcMSLDNonbondedForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
