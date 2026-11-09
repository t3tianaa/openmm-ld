/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "ReferenceMSLDKernelFactory.h"
#include "ReferenceMSLDKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/windowsExport.h"

using namespace MSLDPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    // The CPU platform shares the Reference platform's data layout, so the same kernel serves both.
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (platform.getName() == "Reference" || platform.getName() == "CPU") {
            ReferenceMSLDKernelFactory* factory = new ReferenceMSLDKernelFactory();
            platform.registerKernelFactory(CalcMSLDNonbondedForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerMSLDReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceMSLDKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcMSLDNonbondedForceKernel::Name())
        return new ReferenceCalcMSLDNonbondedForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
