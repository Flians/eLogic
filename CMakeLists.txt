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

add_subdirectory(tools)
add_subdirectory(mig_egg)
add_subdirectory(emig)