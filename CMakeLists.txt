cmake_minimum_required(VERSION 3.15.0...3.21.0)

project(testTriArborX VERSION 0.0.1 LANGUAGES CXX)

find_package(ArborX REQUIRED)
find_package(Omega_h REQUIRED)

add_executable(testTri testTri.cpp)
target_link_libraries(testTri PRIVATE ArborX::ArborX Omega_h::omega_h)
