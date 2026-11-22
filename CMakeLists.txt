#---------------------------------------------------
# OpenMM MSLD (multisite lambda dynamics) plugin
#
# Layout follows openmmexampleplugin:
#   openmmapi/               public API (MSLDNonbondedForce) -> libOpenMMMSLD
#   serialization/           XML serialization proxy           -> libOpenMMMSLD
#   platforms/reference/     Reference/CPU kernels             -> libOpenMMMSLDReference (plugins dir)
#   platforms/common/        GPU kernel (common compute API), built into
#   platforms/opencl|cuda/   -> libOpenMMMSLDOpenCL / libOpenMMMSLDCUDA (plugins dir)
#   python/                  SWIG wrappers (later)
#----------------------------------------------------

cmake_minimum_required(VERSION 3.17)
project(OpenMMMSLD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMM installation: default to the active conda environment.
if(DEFINED ENV{CONDA_PREFIX})
    set(OPENMM_DIR_DEFAULT "$ENV{CONDA_PREFIX}")
else()
    set(OPENMM_DIR_DEFAULT "/usr/local/openmm")
endif()
set(OPENMM_DIR "${OPENMM_DIR_DEFAULT}" CACHE PATH "Where OpenMM is installed")
if(NOT EXISTS "${OPENMM_DIR}/include/OpenMM.h")
    message(FATAL_ERROR "OpenMM headers not found under OPENMM_DIR=${OPENMM_DIR}")
endif()
include_directories("${OPENMM_DIR}/include")
link_directories("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

if(NOT CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH "Where to install the plugin" FORCE)
endif()

set(CMAKE_INSTALL_RPATH "${OPENMM_DIR}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(MSLD_LIBRARY_NAME OpenMMMSLD)
set(SHARED_MSLD_TARGET ${MSLD_LIBRARY_NAME})

# ---- API + serialization library ----
file(GLOB API_SOURCES openmmapi/src/*.cpp serialization/src/*.cpp)
add_library(${SHARED_MSLD_TARGET} SHARED ${API_SOURCES})
target_include_directories(${SHARED_MSLD_TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization/include)
target_link_libraries(${SHARED_MSLD_TARGET} OpenMM)
set_target_properties(${SHARED_MSLD_TARGET} PROPERTIES
    COMPILE_FLAGS "-DMSLD_BUILDING_SHARED_LIBRARY")

install(TARGETS ${SHARED_MSLD_TARGET}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)
file(GLOB API_HEADERS openmmapi/include/*.h)
file(GLOB API_INTERNAL_HEADERS openmmapi/include/internal/*.h)
install(FILES ${API_HEADERS} DESTINATION include)
install(FILES ${API_INTERNAL_HEADERS} DESTINATION include/internal)

# ---- tests ----
enable_testing()
add_subdirectory(platforms/reference)

# ---- GPU platforms (common-platform kernel) ----
find_package(OpenCL QUIET)
option(MSLD_BUILD_OPENCL_LIB "Build the OpenCL kernels" ${OpenCL_FOUND})
if(MSLD_BUILD_OPENCL_LIB)
    find_package(OpenCL REQUIRED)
    add_subdirectory(platforms/opencl)
endif()
# Only cuda.h and the driver library are needed (OpenMM compiles the kernels at run time).
find_package(CUDAToolkit QUIET)
set(_cuda_hints $ENV{CUDA_HOME} $ENV{CUDA_PATH} /usr/local/cuda ${CUDAToolkit_ROOT} ${OPENMM_DIR} ${OPENMM_DIR}/targets/x86_64-linux)
find_path(MSLD_CUDA_INCLUDE_DIR cuda.h HINTS ${CUDAToolkit_INCLUDE_DIRS} PATHS ${_cuda_hints} PATH_SUFFIXES include targets/x86_64-linux/include)
find_library(MSLD_CUDA_DRIVER_LIBRARY cuda HINTS ${CUDAToolkit_LIBRARY_DIR} PATHS ${_cuda_hints}
    PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs targets/x86_64-linux/lib/stubs)
if(MSLD_CUDA_INCLUDE_DIR AND MSLD_CUDA_DRIVER_LIBRARY AND EXISTS "${OPENMM_DIR}/lib/plugins/libOpenMMCUDA.so")
    set(_have_cuda ON)
else()
    set(_have_cuda OFF)
endif()
option(MSLD_BUILD_CUDA_LIB "Build the CUDA kernels" ${_have_cuda})
if(MSLD_BUILD_CUDA_LIB)
    if(NOT MSLD_CUDA_INCLUDE_DIR OR NOT MSLD_CUDA_DRIVER_LIBRARY)
        message(FATAL_ERROR "MSLD_BUILD_CUDA_LIB is ON but cuda.h or libcuda was not found; set CUDA_HOME")
    endif()
    message(STATUS "MSLD CUDA kernels: cuda.h in ${MSLD_CUDA_INCLUDE_DIR}, driver ${MSLD_CUDA_DRIVER_LIBRARY}")
    add_subdirectory(platforms/cuda)
endif()
add_subdirectory(serialization/tests)

option(MSLD_BUILD_PYTHON_WRAPPERS "Build the SWIG Python wrapper (module msldplugin)" ON)
if(MSLD_BUILD_PYTHON_WRAPPERS)
    add_subdirectory(python)
endif()
