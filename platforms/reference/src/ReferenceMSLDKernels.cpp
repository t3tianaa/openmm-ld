/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "ReferenceMSLDKernels.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/reference/ReferencePME.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <algorithm>
#include <cmath>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

static const double KE = ONE_4PI_EPS0;
static const double TWO_OVER_SQRT_PI = 1.1283791670955126;

static ReferencePlatform::PlatformData* getPlatformData(ContextImpl& context) {
    return reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
}

ReferenceCalcMSLDNonbondedForceKernel::~ReferenceCalcMSLDNonbondedForceKernel() {
    if (pme != NULL)
        delete pme;
}

void ReferenceCalcMSLDNonbondedForceKernel::readParameters(const MSLDNonbondedForce& force) {
    charge.resize(numParticles);
    sigma.resize(numParticles);
    epsilon.resize(numParticles);
    ljType.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        int b;
        force.getParticleParameters(i, charge[i], sigma[i], epsilon[i], b);
        if (b != block[i])
            throw OpenMMException("updateParametersInContext: the block of a particle cannot be changed");
        ljType[i] = force.getParticleLJType(i);
    }
    nbfix.clear();
    for (int k = 0; k < force.getNumNBFixPairs(); k++) {
        int t1, t2;
        double s, e;
        force.getNBFixPairParameters(k, t1, t2, s, e);
        nbfix[make_pair(min(t1, t2), max(t1, t2))] = make_pair(s, e);
    }
    set<pair<int, int> > newExclusions;
    exceptions14.clear();
    for (int k = 0; k < force.getNumExceptions(); k++) {
        Exception14 e;
        force.getExceptionParameters(k, e.i, e.j, e.chargeProd, e.sigma, e.epsilon);
        newExclusions.insert(make_pair(min(e.i, e.j), max(e.i, e.j)));
        if (e.chargeProd != 0.0 || e.epsilon != 0.0)
            exceptions14.push_back(e);
    }
    if (!exclusions.empty() && newExclusions != exclusions)
        throw OpenMMException("updateParametersInContext: the set of exceptions has changed");
    exclusions = newExclusions;
}

void ReferenceCalcMSLDNonbondedForceKernel::initialize(const System& system, const MSLDNonbondedForce& force) {
    numParticles = force.getNumParticles();
    numBlocks = force.getNumBlocks();
    blockSite.resize(numBlocks);
    lambdaNames.resize(numBlocks);
    for (int b = 0; b < numBlocks; b++) {
        blockSite[b] = force.getBlockSite(b);
        if (b > 0)
            lambdaNames[b] = MSLDNonbondedForce::getLambdaName(b);
    }
    block.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double q, s, e;
        force.getParticleParameters(i, q, s, e, block[i]);
    }
    exclusions.clear();
    readParameters(force);

    // Excluded pairs whose erf term has to be removed from the PME reciprocal sum: every exception
    // pair, plus every pair between different blocks of the same site.
    pmeExclusions.assign(exclusions.begin(), exclusions.end());
    vector<int> alchemical;
    for (int i = 0; i < numParticles; i++)
        if (block[i] != 0)
            alchemical.push_back(i);
    for (int m = 0; m < (int) alchemical.size(); m++)
        for (int n = m+1; n < (int) alchemical.size(); n++) {
            int i = alchemical[m], j = alchemical[n];
            if (block[i] != block[j] && blockSite[block[i]] == blockSite[block[j]] && exclusions.find(make_pair(i, j)) == exclusions.end())
                pmeExclusions.push_back(make_pair(i, j));
        }

    method = force.getNonbondedMethod();
    periodic = (method == MSLDNonbondedForce::CutoffPeriodic || method == MSLDNonbondedForce::PME);
    cutoff = (method != MSLDNonbondedForce::NoCutoff);
    exceptionsPeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
    cutoffDistance = force.getCutoffDistance();
    useSwitch = cutoff && force.getUseSwitchingFunction();
    switchingDistance = force.getSwitchingDistance();
    softAlpha = force.getSoftcoreAlpha();
    sigmaQ6 = pow(force.getSoftcoreSigmaQ(), 6.0);
    krf = crf = 0.0;
    if (method == MSLDNonbondedForce::CutoffNonPeriodic || method == MSLDNonbondedForce::CutoffPeriodic) {
        double eps = force.getReactionFieldDielectric();
        krf = pow(cutoffDistance, -3.0)*(eps-1.0)/(2.0*eps+1.0);
        crf = (1.0/cutoffDistance)*(3.0*eps)/(2.0*eps+1.0);
    }
    ewaldAlpha = 0.0;
    gridSize[0] = gridSize[1] = gridSize[2] = 0;
    if (method == MSLDNonbondedForce::PME) {
        // Use OpenMM's own parameter selection so alpha and the grid match NonbondedForce exactly.
        NonbondedForce equivalent;
        equivalent.setNonbondedMethod(NonbondedForce::PME);
        equivalent.setCutoffDistance(cutoffDistance);
        equivalent.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
        double alpha;
        int nx, ny, nz;
        force.getPMEParameters(alpha, nx, ny, nz);
        equivalent.setPMEParameters(alpha, nx, ny, nz);
        for (int i = 0; i < numParticles; i++)
            equivalent.addParticle(charge[i], sigma[i], epsilon[i]);
        NonbondedForceImpl::calcPMEParameters(system, equivalent, ewaldAlpha, gridSize[0], gridSize[1], gridSize[2], false);
        pme = new ReferencePME(ewaldAlpha, numParticles, gridSize, 5, 1.0);
    }
}

void ReferenceCalcMSLDNonbondedForceKernel::pairSigmaEpsilon(int i, int j, double& sig, double& eps) const {
    if (ljType[i] >= 0 && ljType[j] >= 0 && !nbfix.empty()) {
        map<pair<int, int>, pair<double, double> >::const_iterator f = nbfix.find(make_pair(min(ljType[i], ljType[j]), max(ljType[i], ljType[j])));
        if (f != nbfix.end()) {
            sig = f->second.first;
            eps = f->second.second;
            return;
        }
    }
    sig = 0.5*(sigma[i]+sigma[j]);
    eps = sqrt(epsilon[i]*epsilon[j]);
}

Vec3 ReferenceCalcMSLDNonbondedForceKernel::delta(const Vec3& xi, const Vec3& xj, const Vec3* box, bool usePeriodic) const {
    Vec3 d = xj-xi;
    if (usePeriodic) {
        d -= box[2]*floor(d[2]/box[2][2]+0.5);
        d -= box[1]*floor(d[1]/box[1][1]+0.5);
        d -= box[0]*floor(d[0]/box[0][0]+0.5);
    }
    return d;
}

void ReferenceCalcMSLDNonbondedForceKernel::coulombKernel(double x, double& g, double& dg) const {
    if (method == MSLDNonbondedForce::NoCutoff) {
        g = 1.0/x;
        dg = -1.0/(x*x);
    }
    else if (method == MSLDNonbondedForce::PME) {
        double erfcx = erfc(ewaldAlpha*x);
        g = erfcx/x;
        dg = -erfcx/(x*x) - TWO_OVER_SQRT_PI*ewaldAlpha*exp(-ewaldAlpha*ewaldAlpha*x*x)/x;
    }
    else {
        g = 1.0/x + krf*x*x - crf;
        dg = -1.0/(x*x) + 2.0*krf*x;
    }
}

double ReferenceCalcMSLDNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ReferencePlatform::PlatformData* data = getPlatformData(context);
    const vector<Vec3>& pos = *data->positions;
    vector<Vec3>& force = *data->forces;
    const Vec3* box = data->periodicBoxVectors;
    if (periodic) {
        double minAllowedSize = 1.999999*cutoffDistance;
        if (box[0][0] < minAllowedSize || box[1][1] < minAllowedSize || box[2][2] < minAllowedSize)
            throw OpenMMException("MSLDNonbondedForce: the periodic box size has decreased to less than twice the nonbonded cutoff.");
    }
    vector<double> lam(numBlocks, 1.0);
    for (int b = 1; b < numBlocks; b++)
        lam[b] = context.getParameter(lambdaNames[b]);
    vector<double> dEdLam(numBlocks, 0.0);
    double energy = 0.0;
    double cutoff2 = cutoffDistance*cutoffDistance;

    // ---- direct-space pairs ----
    for (int i = 0; i < numParticles; i++) {
        int bi = block[i];
        for (int j = i+1; j < numParticles; j++) {
            int bj = block[j];
            if (bi != 0 && bj != 0 && bi != bj && blockSite[bi] == blockSite[bj])
                continue;                                   // different blocks of one site
            if (exclusions.find(make_pair(i, j)) != exclusions.end())
                continue;
            Vec3 d = delta(pos[i], pos[j], box, periodic);
            double r2 = d.dot(d);
            if (cutoff && r2 >= cutoff2)
                continue;
            double r = sqrt(r2);

            // pair coupling L and dL/dlambda
            double L = 1.0, dL1 = 0.0, dL2 = 0.0;
            int b1 = 0, b2 = 0;
            if (bi == 0 && bj == 0) {
            }
            else if (bi == bj || bi == 0 || bj == 0) {
                b1 = (bi != 0 ? bi : bj);
                L = lam[b1];
                dL1 = 1.0;
            }
            else {
                b1 = bi;
                b2 = bj;
                L = lam[bi]*lam[bj];
                dL1 = lam[bj];
                dL2 = lam[bi];
            }
            bool alchemical = (b1 != 0);
            double shift = (alchemical ? softAlpha*(1.0-L) : 0.0);
            double r6 = r2*r2*r2;
            double U = 0.0, dUdr = 0.0, dUdL = 0.0;

            // Lennard-Jones (Beutler soft-core)
            double sig, eps;
            pairSigmaEpsilon(i, j, sig, eps);
            if (eps != 0.0 && sig != 0.0) {
                double sw = 1.0, dsw = 0.0;
                if (useSwitch && r > switchingDistance) {
                    double width = cutoffDistance-switchingDistance;
                    double t = (r-switchingDistance)/width;
                    sw = 1.0+t*t*t*(-10.0+t*(15.0-t*6.0));
                    dsw = t*t*(-30.0+t*(60.0-t*30.0))/width;
                }
                double sig2 = sig*sig;
                double sig6 = sig2*sig2*sig2;
                double s = r6/sig6 + shift;
                double inv = 1.0/s;
                double f = inv*inv - inv;
                double fp = -2.0*inv*inv*inv + inv*inv;
                double ulj = L*4.0*eps*f;
                double duljdr = L*4.0*eps*fp*6.0*r6/(r*sig6);
                U += ulj*sw;
                dUdr += duljdr*sw + ulj*dsw;
                dUdL += 4.0*eps*(f - softAlpha*L*fp)*sw;
            }

            // Coulomb (soft-core on the distance)
            double qq = KE*charge[i]*charge[j];
            if (qq != 0.0) {
                double rsc = (shift > 0.0 ? pow(r6 + shift*sigmaQ6, 1.0/6.0) : r);
                double g, dg;
                coulombKernel(rsc, g, dg);
                double rsc5 = rsc*rsc*rsc*rsc*rsc;
                U += L*qq*g;
                dUdr += L*qq*dg*(r6/r)/rsc5;
                dUdL += qq*(g - L*dg*softAlpha*sigmaQ6/(6.0*rsc5));
            }

            energy += U;
            if (includeForces) {
                Vec3 f = d*(dUdr/r);
                force[i] += f;
                force[j] -= f;
            }
            if (b1 != 0)
                dEdLam[b1] += dUdL*dL1;
            if (b2 != 0)
                dEdLam[b2] += dUdL*dL2;
        }
    }

    // ---- 1-4 exceptions: L (KE qq/r + LJ), no soft-core, no cutoff ----
    for (int k = 0; k < (int) exceptions14.size(); k++) {
        const Exception14& e = exceptions14[k];
        Vec3 d = delta(pos[e.i], pos[e.j], box, periodic && exceptionsPeriodic);
        double r2 = d.dot(d);
        double r = sqrt(r2);
        double u = KE*e.chargeProd/r;
        double dudr = -u/r;
        if (e.epsilon != 0.0) {
            double sr2 = e.sigma*e.sigma/r2;
            double sr6 = sr2*sr2*sr2;
            u += 4.0*e.epsilon*(sr6*sr6-sr6);
            dudr += 4.0*e.epsilon*(-12.0*sr6*sr6+6.0*sr6)/r;
        }
        int b = (block[e.i] != 0 ? block[e.i] : block[e.j]);
        double L = (b == 0 ? 1.0 : lam[b]);
        energy += L*u;
        if (includeForces) {
            Vec3 f = d*(L*dudr/r);
            force[e.i] += f;
            force[e.j] -= f;
        }
        if (b != 0)
            dEdLam[b] += u;
    }

    // ---- PME reciprocal space on charges lambda_b(i) q_i ----
    if (method == MSLDNonbondedForce::PME) {
        vector<double> Q(numParticles);
        for (int i = 0; i < numParticles; i++)
            Q[i] = lam[block[i]]*charge[i];
        vector<Vec3> recipForces(numParticles, Vec3());
        double recipEnergy = 0.0;
        Vec3 boxVectors[3] = {box[0], box[1], box[2]};
        pme->exec(pos, recipForces, Q, boxVectors, recipEnergy);
        energy += recipEnergy;
        if (includeForces)
            for (int i = 0; i < numParticles; i++)
                force[i] += recipForces[i];
        vector<double> dEdQ(numParticles, 0.0);
        bool needDerivs = (numBlocks > 1);
        if (needDerivs) {
            vector<int> indices(numParticles);
            for (int i = 0; i < numParticles; i++)
                indices[i] = i;
            vector<double> recipDerivs(numParticles, 0.0);
            pme->exec_charge_derivatives(pos, recipDerivs, indices, Q, boxVectors);
            for (int i = 0; i < numParticles; i++)
                dEdQ[i] += recipDerivs[i];
        }

        // self energy
        double selfScale = KE*ewaldAlpha/sqrt(M_PI);
        for (int i = 0; i < numParticles; i++) {
            energy -= selfScale*Q[i]*Q[i];
            dEdQ[i] -= 2.0*selfScale*Q[i];
        }

        // neutralizing background for a net charge (as NonbondedForce): -KE pi (sum Q)^2 / (2 V alpha^2)
        double totalCharge = 0.0;
        for (int i = 0; i < numParticles; i++)
            totalCharge += Q[i];
        if (totalCharge != 0.0 || needDerivs) {
            double volume = box[0][0]*box[1][1]*box[2][2];
            double scale = KE*M_PI/(volume*ewaldAlpha*ewaldAlpha);
            energy -= 0.5*scale*totalCharge*totalCharge;
            for (int i = 0; i < numParticles; i++)
                dEdQ[i] -= scale*totalCharge;
        }

        // remove the reciprocal-space interaction of excluded pairs
        for (int k = 0; k < (int) pmeExclusions.size(); k++) {
            int i = pmeExclusions[k].first, j = pmeExclusions[k].second;
            if (Q[i] == 0.0 && Q[j] == 0.0)
                continue;
            Vec3 d = delta(pos[i], pos[j], box, true);
            double r2 = d.dot(d);
            double r = sqrt(r2);
            double erfOverR;
            if (ewaldAlpha*r < 1e-6)
                erfOverR = TWO_OVER_SQRT_PI*ewaldAlpha;
            else {
                double erfr = erf(ewaldAlpha*r);
                erfOverR = erfr/r;
                if (includeForces) {
                    double dEdr = -KE*Q[i]*Q[j]*(TWO_OVER_SQRT_PI*ewaldAlpha*exp(-ewaldAlpha*ewaldAlpha*r2) - erfOverR)/r;
                    Vec3 f = d*(dEdr/r);
                    force[i] += f;
                    force[j] -= f;
                }
            }
            energy -= KE*Q[i]*Q[j]*erfOverR;
            dEdQ[i] -= KE*Q[j]*erfOverR;
            dEdQ[j] -= KE*Q[i]*erfOverR;
        }
        for (int i = 0; i < numParticles; i++)
            if (block[i] != 0)
                dEdLam[block[i]] += charge[i]*dEdQ[i];
    }

    map<string, double>& derivs = *data->energyParameterDerivatives;
    for (int b = 1; b < numBlocks; b++)
        derivs[lambdaNames[b]] += dEdLam[b];
    return energy;
}

void ReferenceCalcMSLDNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const MSLDNonbondedForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: the number of particles has changed");
    if (force.getNumBlocks() != numBlocks)
        throw OpenMMException("updateParametersInContext: the number of blocks has changed");
    for (int b = 1; b < numBlocks; b++)
        if (force.getBlockSite(b) != blockSite[b])
            throw OpenMMException("updateParametersInContext: the site of a block cannot be changed");
    readParameters(force);
}

void ReferenceCalcMSLDNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (method != MSLDNonbondedForce::PME)
        throw OpenMMException("getPMEParametersInContext: this Context is not using PME");
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}
