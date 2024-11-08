set(MOCKTURTLE_EXAMPLES OFF)
set(MOCKTURTLE_EXPERIMENTS OFF)
if(CMAKE_HOST_WIN32)
    add_compile_options("-fpermissive")
endif()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(mockturtle)

add_library(experiments INTERFACE)
target_include_directories(experiments INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/mockturtle/experiments)
target_link_libraries(experiments INTERFACE fmt json)
if (ENABLE_MATPLOTLIB)
  target_link_libraries(experiments INTERFACE matplot)
endif()

add_subdirectory(pybind11)

pybind11_add_module(MIGPy ${CMAKE_CURRENT_SOURCE_DIR}/mig_syn.cpp)
target_link_libraries(MIGPy PUBLIC mockturtle)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
# add_subdirectory(symengine)