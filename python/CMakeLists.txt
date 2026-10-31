#---------------------------------------------------
# SWIG Python wrapper: module "msldplugin"
#---------------------------------------------------

find_package(SWIG 4.0 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

set(WRAP_FILE ${CMAKE_CURRENT_BINARY_DIR}/MSLDPluginWrapper.cpp)
set(SWIG_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/msldplugin.i)

add_custom_command(
    OUTPUT ${WRAP_FILE} ${CMAKE_CURRENT_BINARY_DIR}/msldplugin.py
    COMMAND ${SWIG_EXECUTABLE} -python -c++
        -outdir ${CMAKE_CURRENT_BINARY_DIR}
        -o ${WRAP_FILE}
        -I${OPENMM_DIR}/include
        -I${CMAKE_SOURCE_DIR}/openmmapi/include
        ${SWIG_INPUT}
    DEPENDS ${SWIG_INPUT} ${CMAKE_SOURCE_DIR}/openmmapi/include/MSLDNonbondedForce.h
    COMMENT "Generating the msldplugin SWIG wrapper")

Python3_add_library(_msldplugin MODULE WITH_SOABI ${WRAP_FILE})
target_link_libraries(_msldplugin PRIVATE ${SHARED_MSLD_TARGET} OpenMM)
set_target_properties(_msldplugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    BUILD_RPATH "${CMAKE_BINARY_DIR};${OPENMM_DIR}/lib"
    INSTALL_RPATH "${OPENMM_DIR}/lib")
# SWIG-generated code triggers a few harmless warnings.
target_compile_options(_msldplugin PRIVATE -Wno-deprecated-declarations)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS _msldplugin LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/msldplugin.py DESTINATION ${PYTHON_SITE_PACKAGES})

# pytest suite (needs numpy and pytest in the Python environment). The comparison with the
# OpenMM-MSLD Python engine (tests/test_python_engine.py) runs only if MSLD_ENGINE_PATH is set.
set(MSLD_ENGINE_PATH "" CACHE PATH "OpenMM-MSLD checkout (the directory containing msld/) for the engine comparison tests")
add_test(NAME PythonTests
    COMMAND ${Python3_EXECUTABLE} -m pytest -q -rs ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set_tests_properties(PythonTests PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};MSLD_REFERENCE_PLUGIN=$<TARGET_FILE:OpenMMMSLDReference>;MSLD_ENGINE_PATH=${MSLD_ENGINE_PATH}")
