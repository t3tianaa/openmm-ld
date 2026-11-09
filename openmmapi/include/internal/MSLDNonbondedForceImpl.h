#ifndef OPENMM_MSLDNONBONDEDFORCEIMPL_H_
#define OPENMM_MSLDNONBONDEDFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "MSLDNonbondedForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <vector>

namespace MSLDPlugin {

/**
 * The internal implementation of MSLDNonbondedForce.
 */
class OPENMM_EXPORT_MSLD MSLDNonbondedForceImpl : public OpenMM::ForceImpl {
public:
    MSLDNonbondedForceImpl(const MSLDNonbondedForce& owner);
    ~MSLDNonbondedForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const MSLDNonbondedForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters();
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Check a force for consistency (block/site indices, illegal exceptions). Throws on error.
     */
    static void validate(const OpenMM::System& system, const MSLDNonbondedForce& force);
private:
    const MSLDNonbondedForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace MSLDPlugin

#endif /*OPENMM_MSLDNONBONDEDFORCEIMPL_H_*/
