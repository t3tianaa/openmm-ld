#---------------------------------------------------
# Reference (and CPU) platform kernels for the MSLD plugin
#---------------------------------------------------

set(REFERENCE_TARGET OpenMMMSLDReference)

file(GLOB REFERENCE_SOURCES src/*.cpp)
add_library(${REFERENCE_TARGET} SHARED ${REFERENCE_SOURCES})
target_include_directories(${REFERENCE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${REFERENCE_TARGET} OpenMM ${SHARED_MSLD_TARGET})
set_target_properties(${REFERENCE_TARGET} PROPERTIES
    COMPILE_FLAGS "-DMSLD_BUILDING_SHARED_LIBRARY")

install(TARGETS ${REFERENCE_TARGET}
    LIBRARY DESTINATION lib/plugins
    RUNTIME DESTINATION lib/plugins)

# ---- tests ----
file(GLOB TEST_PROGS tests/*.cpp)
foreach(TEST_PROG ${TEST_PROGS})
    get_filename_component(TEST_ROOT ${TEST_PROG} NAME_WE)
    add_executable(${TEST_ROOT} ${TEST_PROG})
    target_link_libraries(${TEST_ROOT} ${SHARED_MSLD_TARGET} ${REFERENCE_TARGET})
    target_compile_definitions(${TEST_ROOT} PRIVATE OPENMM_PLUGIN_DIR="${OPENMM_DIR}/lib/plugins")
    add_test(NAME ${TEST_ROOT} COMMAND ${TEST_ROOT})
endforeach()
