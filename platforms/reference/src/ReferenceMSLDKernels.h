#ifndef REFERENCE_MSLD_KERNELS_H_
#define REFERENCE_MSLD_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "MSLDKernels.h"
#include "openmm/Platform.h"
#include "openmm/Vec3.h"
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace OpenMM {
class ReferencePME;
}

namespace MSLDPlugin {

/**
 * Reference implementation of MSLDNonbondedForce (also registered for the CPU platform).
 *
 * Direct space is a plain O(N^2) pair loop. With PME the reciprocal part is OpenMM's own
 * ReferencePME on the charges lambda_b(i) q_i, with the self term and the excluded-pair erf terms
 * subtracted, so at lambda = 1 it matches OpenMM::NonbondedForce exactly.
 */
class ReferenceCalcMSLDNonbondedForceKernel : public CalcMSLDNonbondedForceKernel {
public:
    ReferenceCalcMSLDNonbondedForceKernel(std::string name, const OpenMM::Platform& platform) : CalcMSLDNonbondedForceKernel(name, platform), pme(NULL) {
    }
    ~ReferenceCalcMSLDNonbondedForceKernel();
    void initialize(const OpenMM::System& system, const MSLDNonbondedForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const MSLDNonbondedForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    struct Exception14 {
        int i, j;
        double chargeProd, sigma, epsilon;
    };
    void readParameters(const MSLDNonbondedForce& force);
    void pairSigmaEpsilon(int i, int j, double& sigma, double& epsilon) const;
    OpenMM::Vec3 delta(const OpenMM::Vec3& xi, const OpenMM::Vec3& xj, const OpenMM::Vec3* box, bool periodic) const;
    void coulombKernel(double r, double& g, double& dg) const;

    int numParticles, numBlocks;
    MSLDNonbondedForce::NonbondedMethod method;
    bool periodic, cutoff, useSwitch, exceptionsPeriodic;
    double cutoffDistance, switchingDistance, krf, crf, ewaldAlpha, softAlpha, sigmaQ6;
    int gridSize[3];
    std::vector<double> charge, sigma, epsilon;
    std::vector<int> block, blockSite, ljType;
    std::vector<std::string> lambdaNames;
    std::map<std::pair<int, int>, std::pair<double, double> > nbfix;
    std::set<std::pair<int, int> > exclusions;            // all exception pairs (i < j)
    std::vector<std::pair<int, int> > pmeExclusions;      // exceptions + same-site cross-block pairs
    std::vector<Exception14> exceptions14;
    OpenMM::ReferencePME* pme;
};

} // namespace MSLDPlugin

#endif /*REFERENCE_MSLD_KERNELS_H_*/
