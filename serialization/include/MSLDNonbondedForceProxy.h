#ifndef OPENMM_MSLD_NONBONDED_FORCE_PROXY_H_
#define OPENMM_MSLD_NONBONDED_FORCE_PROXY_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportMSLD.h"
#include "openmm/serialization/SerializationProxy.h"

namespace MSLDPlugin {

/**
 * This is a proxy for serializing MSLDNonbondedForce objects.
 */
class OPENMM_EXPORT_MSLD MSLDNonbondedForceProxy : public OpenMM::SerializationProxy {
public:
    MSLDNonbondedForceProxy();
    void serialize(const void* object, OpenMM::SerializationNode& node) const;
    void* deserialize(const OpenMM::SerializationNode& node) const;
};

} // namespace MSLDPlugin

#endif /*OPENMM_MSLD_NONBONDED_FORCE_PROXY_H_*/
