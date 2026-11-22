#ifndef COMMON_MSLD_KERNELS_H_
#define COMMON_MSLD_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 *                                                                            *
 * GPU (CUDA / OpenCL) implementation of MSLDNonbondedForce, written against  *
 * OpenMM's common compute platform.  The PME reciprocal-space kernels are    *
 * adapted from OpenMM's platforms/common/src/kernels/pme.cc (LGPL v3).       *
 * -------------------------------------------------------------------------- */

#include "MSLDKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/FFT3D.h"
#include <string>
#include <vector>

namespace MSLDPlugin {

class CommonCalcMSLDNonbondedForceKernel : public CalcMSLDNonbondedForceKernel {
public:
    CommonCalcMSLDNonbondedForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc) :
            CalcMSLDNonbondedForceKernel(name, platform), cc(cc), hasInitializedKernels(false), info(NULL) {
    }
    ~CommonCalcMSLDNonbondedForceKernel();
    void initialize(const OpenMM::System& system, const MSLDNonbondedForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const MSLDNonbondedForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class ForceInfo;
    void uploadParameters(const MSLDNonbondedForce& force);
    void initializeKernels();
    std::string addDerivative(const std::vector<std::string>& derivVariables, const std::string& blockVar, const std::string& value) const;

    OpenMM::ComputeContext& cc;
    bool hasInitializedKernels, usePME, useFixedPointChargeSpreading;
    ForceInfo* info;
    int numParticles, numBlocks, numTypes, numExceptions14;
    MSLDNonbondedForce::NonbondedMethod method;
    std::vector<int> block, blockSite;
    std::vector<std::string> lambdaNames;
    std::vector<int> lambdaGlobalIndex;
    std::vector<std::pair<int, int> > exclusionPairs, pmeExclusions;
    std::vector<int> exception14Index;

    // Device arrays
    OpenMM::ComputeArray particleParams;     // float4: charge, sigma, sqrt(epsilon), block + 1024*(ljType+1)
    OpenMM::ComputeArray nbfixTable;         // float2: sigma, epsilon per type pair, sigma < 0 means "combining rules"
    OpenMM::ComputeArray lambdaIndex;        // int: index of each block's lambda in the context's global parameter array
    OpenMM::ComputeArray exceptionParams;    // float4: KE*chargeProd, sigma, 4*epsilon, block

    // PME
    double ewaldAlpha;
    int gridSize[3];
    OpenMM::ComputeArray charges, pmeGrid1, pmeGrid2, moduliX, moduliY, moduliZ, derivIndex;
    OpenMM::FFT3D fft;
    OpenMM::ComputeKernel chargesKernel, spreadKernel, finishSpreadKernel, convolutionKernel, energyKernel, interpolateKernel;
    std::vector<double> blockChargeSum, blockChargeSquaredSum;
};

} // namespace MSLDPlugin

#endif /*COMMON_MSLD_KERNELS_H_*/
