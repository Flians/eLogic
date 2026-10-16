# Keep the ePMapping library optimization in this repository rather than a fork.
execute_process(
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/apply_mockturtle_patches.sh"
  RESULT_VARIABLE EPM_PATCH_RESULT
)
if(NOT EPM_PATCH_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not apply the ePMapping mockturtle patch; resolve the reported conflict first.")
endif()

add_library(epmapping INTERFACE)

target_include_directories(epmapping INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../tools/mockturtle/include
  ${CMAKE_CURRENT_BINARY_DIR}/../map_egg/cxxbridge
  ${CMAKE_CURRENT_BINARY_DIR}/../map_egg
)

target_link_libraries(epmapping INTERFACE map_egg mockturtle experiments)
add_dependencies(epmapping map_egg_lib)

add_executable(epmapping_check main.cpp)
target_link_libraries(epmapping_check PRIVATE epmapping)
add_dependencies(epmapping_check map_egg_lib)
