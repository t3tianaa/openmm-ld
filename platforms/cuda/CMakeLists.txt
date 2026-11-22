#---------------------------------------------------
# CUDA kernels for the MSLD plugin (common-platform code + CUDA factory).
# Kernels are compiled at run time by OpenMM, so only the CUDA headers and
# driver library are needed here, not nvcc.
#---------------------------------------------------

set(CUDA_TARGET OpenMMMSLDCUDA)
add_library(${CUDA_TARGET} SHARED
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonMSLDKernels.cpp
    src/CudaMSLDKernelFactory.cpp)
target_include_directories(${CUDA_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/platforms/common/src ${MSLD_CUDA_INCLUDE_DIR})
target_link_libraries(${CUDA_TARGET} OpenMM OpenMMCUDA ${MSLD_CUDA_DRIVER_LIBRARY} ${SHARED_MSLD_TARGET})
set_target_properties(${CUDA_TARGET} PROPERTIES
    COMPILE_FLAGS "-DMSLD_BUILDING_SHARED_LIBRARY"
    INSTALL_RPATH "${OPENMM_DIR}/lib;${OPENMM_DIR}/lib/plugins")
install(TARGETS ${CUDA_TARGET}
    LIBRARY DESTINATION lib/plugins
    RUNTIME DESTINATION lib/plugins)
