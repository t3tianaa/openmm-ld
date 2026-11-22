/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 *                                                                            *
 * Registers the common-platform MSLDNonbondedForce kernel with the           *
 * CUDA platform.                                                          *
 * -------------------------------------------------------------------------- */

#include "CommonMSLDKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"

using namespace MSLDPlugin;
using namespace OpenMM;

namespace {
class CudaMSLDKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        CudaContext& cc = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
        if (name == CalcMSLDNonbondedForceKernel::Name())
            return new CommonCalcMSLDNonbondedForceKernel(name, platform, cc);
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        platform.registerKernelFactory(CalcMSLDNonbondedForceKernel::Name(), new CudaMSLDKernelFactory());
    }
    catch (std::exception& ex) {
        // The CUDA platform is not available.
    }
}
