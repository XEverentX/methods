cmake_minimum_required(VERSION 3.10)
project(optimization_lib)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(optimization_lib src/gradient_descent.cpp include/gradient_descent.hpp)