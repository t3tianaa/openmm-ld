#ifndef OPENMM_WINDOWSEXPORTMSLD_H_
#define OPENMM_WINDOWSEXPORTMSLD_H_

/*
 * Shared libraries are messy in Visual Studio. We have to distinguish three
 * cases:
 *   (1) this header is being used to build the OpenMM MSLD shared library
 *       (dllexport)
 *   (2) this header is being used by a *client* of the shared library
 *       (dllimport)
 *   (3) we are building or using a static library (no decoration)
 */

#ifdef _MSC_VER
    #if defined(MSLD_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_MSLD __declspec(dllexport)
    #elif defined(MSLD_BUILDING_STATIC_LIBRARY) || defined(MSLD_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_MSLD
    #else
        #define OPENMM_EXPORT_MSLD __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_MSLD __attribute__((visibility("default")))
#endif

#endif // OPENMM_WINDOWSEXPORTMSLD_H_
