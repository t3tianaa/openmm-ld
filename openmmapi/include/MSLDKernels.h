#ifndef MSLD_KERNELS_H_
#define MSLD_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "MSLDNonbondedForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace MSLDPlugin {

/**
 * This kernel is invoked by MSLDNonbondedForce to calculate the forces acting on the system,
 * the energy, and dU/dlambda_b for every block.
 */
class CalcMSLDNonbondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcMSLDNonbondedForce";
    }
    CalcMSLDNonbondedForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     */
    virtual void initialize(const OpenMM::System& system, const MSLDNonbondedForce& force) = 0;
    /**
     * Execute the kernel. Returns the potential energy due to the force.
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const MSLDNonbondedForce& force) = 0;
    /**
     * Get the parameters being used for PME.
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

} // namespace MSLDPlugin

#endif /*MSLD_KERNELS_H_*/
