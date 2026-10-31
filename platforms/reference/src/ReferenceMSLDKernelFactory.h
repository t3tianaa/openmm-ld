#ifndef OPENMM_REFERENCEMSLDKERNELFACTORY_H_
#define OPENMM_REFERENCEMSLDKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace MSLDPlugin {

/**
 * This KernelFactory creates the Reference MSLD kernels (used by both the Reference and CPU
 * platforms).
 */
class ReferenceMSLDKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace MSLDPlugin

#endif /*OPENMM_REFERENCEMSLDKERNELFACTORY_H_*/
