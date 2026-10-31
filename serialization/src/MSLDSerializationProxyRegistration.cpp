/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "MSLDNonbondedForce.h"
#include "MSLDNonbondedForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_MSLD void registerMSLDSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerMSLDSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerMSLDSerializationProxies();
#endif

using namespace MSLDPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_MSLD void registerMSLDSerializationProxies() {
    SerializationProxy::registerProxy(typeid(MSLDNonbondedForce), new MSLDNonbondedForceProxy());
}
