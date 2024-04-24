set(MOCKTURTLE_EXAMPLES OFF)
set(MOCKTURTLE_EXPERIMENTS OFF)
if(CMAKE_HOST_WIN32)
    add_compile_options("-fpermissive")
endif()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(mockturtle)

add_subdirectory(pybind11)

pybind11_add_module(MIGPy ${CMAKE_CURRENT_SOURCE_DIR}/mig_syn.cpp)
target_link_libraries(MIGPy PUBLIC mockturtle)
