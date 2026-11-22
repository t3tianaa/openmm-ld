#---------------------------------------------------
# OpenCL kernels for the MSLD plugin (common-platform code + OpenCL factory)
#---------------------------------------------------

set(OPENCL_TARGET OpenMMMSLDOpenCL)
add_library(${OPENCL_TARGET} SHARED
    ${CMAKE_SOURCE_DIR}/platforms/common/src/CommonMSLDKernels.cpp
    src/OpenCLMSLDKernelFactory.cpp)
target_include_directories(${OPENCL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/platforms/common/src ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${OPENCL_TARGET} OpenMM OpenMMOpenCL ${OpenCL_LIBRARIES} ${SHARED_MSLD_TARGET})
set_target_properties(${OPENCL_TARGET} PROPERTIES
    COMPILE_FLAGS "-DMSLD_BUILDING_SHARED_LIBRARY"
    INSTALL_RPATH "${OPENMM_DIR}/lib;${OPENMM_DIR}/lib/plugins")
install(TARGETS ${OPENCL_TARGET}
    LIBRARY DESTINATION lib/plugins
    RUNTIME DESTINATION lib/plugins)
