/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 *                                                                            *
 * Registers the common-platform MSLDNonbondedForce kernel with the           *
 * OpenCL platform.                                                          *
 * -------------------------------------------------------------------------- */

#include "CommonMSLDKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLPlatform.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"

using namespace MSLDPlugin;
using namespace OpenMM;

namespace {
class OpenCLMSLDKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        OpenCLContext& cc = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
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
        Platform& platform = Platform::getPlatformByName("OpenCL");
        platform.registerKernelFactory(CalcMSLDNonbondedForceKernel::Name(), new OpenCLMSLDKernelFactory());
    }
    catch (std::exception& ex) {
        // The OpenCL platform is not available.
    }
}
