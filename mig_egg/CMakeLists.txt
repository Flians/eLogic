# Generate and compile a library.
find_package(PkgConfig REQUIRED)
pkg_check_modules(CBC REQUIRED cbc IMPORTED_TARGET GLOBAL)
add_library(Coin::Cbc ALIAS PkgConfig::CBC)
add_library(Coin::CbcSolver ALIAS PkgConfig::CBC)
pkg_check_modules(OSI_CBC REQUIRED osi-cbc IMPORTED_TARGET GLOBAL)
add_library(Coin::OsiCbc ALIAS PkgConfig::OSI_CBC)
message(STATUS "${CBC_LIBRARIES}")

set(CARGO_CMD cargo build --release)
set(TARGET_DIR "release")
set(RUST_GEN_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libmig_egg.a")
set(RUST_GEN_CXX "${CMAKE_CURRENT_BINARY_DIR}//cxxbridge/mig_egg/src/lib.rs.cc")

add_custom_command(
  OUTPUT ${RUST_GEN_CXX}
  COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} ${CARGO_CMD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(mig_egg STATIC ${RUST_GEN_CXX})
target_link_libraries(mig_egg pthread dl Coin::Cbc ${RUST_GEN_LIB})
target_include_directories(mig_egg SYSTEM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/ ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/)
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/
)
