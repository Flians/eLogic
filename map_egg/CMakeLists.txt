# Generate and compile a library.
find_package(PkgConfig REQUIRED)
pkg_check_modules(CBC REQUIRED cbc IMPORTED_TARGET GLOBAL)
add_library(Coin::Cbc ALIAS PkgConfig::CBC)
add_library(Coin::CbcSolver ALIAS PkgConfig::CBC)
pkg_check_modules(OSI_CBC REQUIRED osi-cbc IMPORTED_TARGET GLOBAL)
add_library(Coin::OsiCbc ALIAS PkgConfig::OSI_CBC)
message(STATUS "${CBC_LIBRARIES}")

set(CARGO_CMD cargo build --release --features ilp-cbc)
set(TARGET_DIR "release")
set(RUST_GEN_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libmap_egg.a")

add_custom_command(
  OUTPUT ${RUST_GEN_LIB}
  COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} ${CARGO_CMD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    ${CMAKE_CURRENT_SOURCE_DIR}/build.rs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs
)
add_custom_target(map_egg_lib ALL DEPENDS ${RUST_GEN_LIB})

add_library(map_egg INTERFACE)
target_link_libraries(map_egg INTERFACE pthread dl Coin::Cbc ${RUST_GEN_LIB})
target_include_directories(map_egg SYSTEM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/ ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/)
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/
)
add_dependencies(map_egg map_egg_lib)