cmake_minimum_required(VERSION 3.16)

project(MIG)

add_definitions(-fPIC)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_USE_RELATIVE_PATHS TRUE)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -ggdb")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -DNDEBUG")
endif()

set(REGG_FOLDER_TO_DELETE "${CMAKE_CURRENT_BINARY_DIR}/mig_egg")
if(EXISTS "${REGG_FOLDER_TO_DELETE}")
    file(REMOVE_RECURSE "${REGG_FOLDER_TO_DELETE}")
    message(STATUS "Deleted folder: ${REGG_FOLDER_TO_DELETE}")
else()
    message(STATUS "Folder does not exist: ${REGG_FOLDER_TO_DELETE}")
endif()

# Homebrew installs the Coin-OR dependencies of CBC (clp, osi, cgl, coinutils)
# as keg-only formulae, so their *.pc files are not linked into the default
# pkg-config search path. cbc.pc declares them in `Requires:`, and CMake's
# pkg_check_modules runs pkg-config with --print-errors, which then fails with
# "Package 'clp', required by 'cbc', not found". Extend PKG_CONFIG_PATH so that
# map_egg/CMakeLists.txt configures out of the box.
if(APPLE)
  foreach(_coin_keg clp osi cgl coinutils)
    set(_coin_pc_dir "/opt/homebrew/opt/${_coin_keg}/lib/pkgconfig")
    if(EXISTS "${_coin_pc_dir}")
      set(ENV{PKG_CONFIG_PATH} "${_coin_pc_dir}:$ENV{PKG_CONFIG_PATH}")
    endif()
  endforeach()
  unset(_coin_pc_dir)
  unset(_coin_keg)
endif()

add_subdirectory(tools)
add_subdirectory(map_egg)
add_subdirectory(epmapping)
#add_subdirectory(emig)
#add_subdirectory(mig_egg)
#add_subdirectory(emapper)