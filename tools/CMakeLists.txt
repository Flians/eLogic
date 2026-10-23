option(MIGBALANCE_APPLY_MOCKTURTLE_OVERLAY "Install the fixed project mockturtle headers before building" ON)
if(MIGBALANCE_APPLY_MOCKTURTLE_OVERLAY)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/apply_mockturtle_overlay.py"
        RESULT_VARIABLE MOCKTURTLE_OVERLAY_RESULT
    )
    if(NOT MOCKTURTLE_OVERLAY_RESULT EQUAL 0)
        message(FATAL_ERROR "Cannot install project mockturtle headers; existing edits were preserved.")
    endif()
    add_custom_target(mockturtle_project_overlay
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/apply_mockturtle_overlay.py"
        COMMENT "Verify/apply fixed project mockturtle headers"
        VERBATIM
    )
endif()

set(MOCKTURTLE_EXAMPLES OFF)
set(MOCKTURTLE_BUILD_EXPERIMENTS ON)
if(CMAKE_HOST_WIN32)
    add_compile_options("-fpermissive")
endif()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(mockturtle)
if(MIGBALANCE_APPLY_MOCKTURTLE_OVERLAY)
    add_dependencies(mockturtle mockturtle_project_overlay)
endif()

add_subdirectory(pybind11)

pybind11_add_module(MIGPy ${CMAKE_CURRENT_SOURCE_DIR}/mig_syn.cpp)
target_link_libraries(MIGPy PUBLIC mockturtle)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
# add_subdirectory(symengine)
