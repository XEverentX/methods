cmake_minimum_required(VERSION 3.10)
project(methods_tests)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/lib/include)

add_executable(methods_tests main.cpp)

target_link_libraries(methods_tests ${CMAKE_SOURCE_DIR}/cmake-build-debug/lib/liboptimization_lib.a)