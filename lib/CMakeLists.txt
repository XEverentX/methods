cmake_minimum_required(VERSION 3.9.2)
project(optimization_lib)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(optimization_lib src/const_gradient_descent.cpp include/gradient_descent.hpp src/fractional_gradient_descent.cpp)