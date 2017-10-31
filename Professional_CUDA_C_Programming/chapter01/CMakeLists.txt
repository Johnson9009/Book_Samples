cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(chapter01 LANGUAGES CXX CUDA)

include_directories("../common")
add_executable(hello hello.cu)
