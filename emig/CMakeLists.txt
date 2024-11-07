set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package (Python 3.11 COMPONENTS Interpreter Development REQUIRED)

add_executable(emig ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(emig PUBLIC mig_egg mockturtle experiments fmt ${Python_LIBRARIES})
target_include_directories(emig PRIVATE ${Python_INCLUDE_DIRS})
add_dependencies(emig mig_egg mockturtle Python::Python)

