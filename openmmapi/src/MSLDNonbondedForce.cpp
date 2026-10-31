/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "MSLDNonbondedForce.h"
#include "internal/MSLDNonbondedForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <algorithm>
#include <sstream>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

MSLDNonbondedForce::MSLDNonbondedForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), switchingDistance(-1.0),
        rfDielectric(78.3), ewaldErrorTol(5e-4), alpha(0.0), softcoreAlpha(0.5), softcoreSigmaQ(0.3),
        useSwitchingFunction(false), exceptionsUsePeriodic(false), nx(0), ny(0), nz(0) {
    blockSite.push_back(0);            // block 0: the environment
    blockDefaultLambda.push_back(1.0);
}

string MSLDNonbondedForce::getLambdaName(int block) {
    stringstream name;
    name << "lambda" << block;
    return name.str();
}

int MSLDNonbondedForce::addBlock(int site, double defaultLambda) {
    if (site < 1)
        throw OpenMMException("MSLDNonbondedForce: a block's site must be >= 1 (site 0 is the environment)");
    blockSite.push_back(site);
    blockDefaultLambda.push_back(defaultLambda);
    return blockSite.size()-1;
}

int MSLDNonbondedForce::getBlockSite(int block) const {
    ASSERT_VALID_INDEX(block, blockSite);
    return blockSite[block];
}

void MSLDNonbondedForce::setBlockSite(int block, int site) {
    ASSERT_VALID_INDEX(block, blockSite);
    if (block == 0)
        throw OpenMMException("MSLDNonbondedForce: block 0 is the environment and has no site");
    if (site < 1)
        throw OpenMMException("MSLDNonbondedForce: a block's site must be >= 1");
    blockSite[block] = site;
}

double MSLDNonbondedForce::getBlockDefaultLambda(int block) const {
    ASSERT_VALID_INDEX(block, blockSite);
    return blockDefaultLambda[block];
}

void MSLDNonbondedForce::setBlockDefaultLambda(int block, double defaultLambda) {
    ASSERT_VALID_INDEX(block, blockSite);
    if (block == 0)
        throw OpenMMException("MSLDNonbondedForce: block 0 is the environment and has no lambda");
    blockDefaultLambda[block] = defaultLambda;
}

int MSLDNonbondedForce::addParticle(double charge, double sigma, double epsilon, int block) {
    ParticleInfo p = {charge, sigma, epsilon, block, -1};
    particles.push_back(p);
    return particles.size()-1;
}

void MSLDNonbondedForce::getParticleParameters(int index, double& charge, double& sigma, double& epsilon, int& block) const {
    ASSERT_VALID_INDEX(index, particles);
    const ParticleInfo& p = particles[index];
    charge = p.charge;
    sigma = p.sigma;
    epsilon = p.epsilon;
    block = p.block;
}

void MSLDNonbondedForce::setParticleParameters(int index, double charge, double sigma, double epsilon, int block) {
    ASSERT_VALID_INDEX(index, particles);
    ParticleInfo& p = particles[index];
    p.charge = charge;
    p.sigma = sigma;
    p.epsilon = epsilon;
    p.block = block;
}

void MSLDNonbondedForce::setParticleLJType(int index, int type) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].ljType = type;
}

int MSLDNonbondedForce::getParticleLJType(int index) const {
    ASSERT_VALID_INDEX(index, particles);
    return particles[index].ljType;
}

int MSLDNonbondedForce::addNBFixPair(int type1, int type2, double sigma, double epsilon) {
    NBFixInfo f = {type1, type2, sigma, epsilon};
    nbfix.push_back(f);
    return nbfix.size()-1;
}

void MSLDNonbondedForce::getNBFixPairParameters(int index, int& type1, int& type2, double& sigma, double& epsilon) const {
    ASSERT_VALID_INDEX(index, nbfix);
    type1 = nbfix[index].type1;
    type2 = nbfix[index].type2;
    sigma = nbfix[index].sigma;
    epsilon = nbfix[index].epsilon;
}

void MSLDNonbondedForce::setNBFixPairParameters(int index, int type1, int type2, double sigma, double epsilon) {
    ASSERT_VALID_INDEX(index, nbfix);
    NBFixInfo f = {type1, type2, sigma, epsilon};
    nbfix[index] = f;
}

int MSLDNonbondedForce::addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace) {
    pair<int, int> key(min(particle1, particle2), max(particle1, particle2));
    map<pair<int, int>, int>::iterator existing = exceptionMap.find(key);
    ExceptionInfo e = {particle1, particle2, chargeProd, sigma, epsilon};
    if (existing != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "MSLDNonbondedForce: there is already an exception for particles " << particle1 << " and " << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[existing->second] = e;
        return existing->second;
    }
    exceptions.push_back(e);
    exceptionMap[key] = exceptions.size()-1;
    return exceptions.size()-1;
}

void MSLDNonbondedForce::getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const {
    ASSERT_VALID_INDEX(index, exceptions);
    const ExceptionInfo& e = exceptions[index];
    particle1 = e.particle1;
    particle2 = e.particle2;
    chargeProd = e.chargeProd;
    sigma = e.sigma;
    epsilon = e.epsilon;
}

void MSLDNonbondedForce::setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon) {
    ASSERT_VALID_INDEX(index, exceptions);
    ExceptionInfo& e = exceptions[index];
    pair<int, int> oldKey(min(e.particle1, e.particle2), max(e.particle1, e.particle2));
    pair<int, int> newKey(min(particle1, particle2), max(particle1, particle2));
    if (oldKey != newKey) {
        if (exceptionMap.find(newKey) != exceptionMap.end())
            throw OpenMMException("MSLDNonbondedForce: there is already an exception for that pair");
        exceptionMap.erase(oldKey);
        exceptionMap[newKey] = index;
    }
    ExceptionInfo updated = {particle1, particle2, chargeProd, sigma, epsilon};
    e = updated;
}

MSLDNonbondedForce::NonbondedMethod MSLDNonbondedForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void MSLDNonbondedForce::setNonbondedMethod(NonbondedMethod method) {
    if (method != NoCutoff && method != CutoffNonPeriodic && method != CutoffPeriodic && method != PME)
        throw OpenMMException("MSLDNonbondedForce: illegal value for nonbonded method");
    nonbondedMethod = method;
}

double MSLDNonbondedForce::getCutoffDistance() const {
    return cutoffDistance;
}

void MSLDNonbondedForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

bool MSLDNonbondedForce::getUseSwitchingFunction() const {
    return useSwitchingFunction;
}

void MSLDNonbondedForce::setUseSwitchingFunction(bool use) {
    useSwitchingFunction = use;
}

double MSLDNonbondedForce::getSwitchingDistance() const {
    return switchingDistance;
}

void MSLDNonbondedForce::setSwitchingDistance(double distance) {
    switchingDistance = distance;
}

double MSLDNonbondedForce::getReactionFieldDielectric() const {
    return rfDielectric;
}

void MSLDNonbondedForce::setReactionFieldDielectric(double dielectric) {
    rfDielectric = dielectric;
}

double MSLDNonbondedForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void MSLDNonbondedForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

void MSLDNonbondedForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void MSLDNonbondedForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

void MSLDNonbondedForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MSLDNonbondedForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

bool MSLDNonbondedForce::getExceptionsUsePeriodicBoundaryConditions() const {
    return exceptionsUsePeriodic;
}

void MSLDNonbondedForce::setExceptionsUsePeriodicBoundaryConditions(bool periodic) {
    exceptionsUsePeriodic = periodic;
}

double MSLDNonbondedForce::getSoftcoreAlpha() const {
    return softcoreAlpha;
}

void MSLDNonbondedForce::setSoftcoreAlpha(double alpha) {
    if (alpha < 0)
        throw OpenMMException("MSLDNonbondedForce: soft-core alpha must be >= 0");
    softcoreAlpha = alpha;
}

double MSLDNonbondedForce::getSoftcoreSigmaQ() const {
    return softcoreSigmaQ;
}

void MSLDNonbondedForce::setSoftcoreSigmaQ(double sigmaQ) {
    if (sigmaQ < 0)
        throw OpenMMException("MSLDNonbondedForce: soft-core sigma_q must be >= 0");
    softcoreSigmaQ = sigmaQ;
}

void MSLDNonbondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<MSLDNonbondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

ForceImpl* MSLDNonbondedForce::createImpl() const {
    return new MSLDNonbondedForceImpl(*this);
}
