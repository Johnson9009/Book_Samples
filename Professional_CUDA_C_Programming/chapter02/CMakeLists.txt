cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(chapter02 LANGUAGES C CXX CUDA)

include_directories("../common")

add_executable(sumArraysOnHost sumArraysOnHost.c)
target_compile_options(sumArraysOnHost PRIVATE -std=c99)

add_executable(checkDimension checkDimension.cu)
target_compile_options(checkDimension PRIVATE -arch=sm_30)

add_executable(defineGridBlock defineGridBlock.cu)
add_executable(addvector sumArraysOnGPU-small-case.cu)
add_executable(sumArraysOnGPU-timer sumArraysOnGPU-timer.cu)
add_executable(checkIndex checkThreadIndex.cu)
add_executable(matrix2D sumMatrixOnGPU-2D-grid-2D-block.cu)
add_executable(matrix1D sumMatrixOnGPU-1D-grid-1D-block.cu)
add_executable(matrix2D1D sumMatrixOnGPU-2D-grid-1D-block.cu)
add_executable(checkDeviceInfor checkDeviceInfor.cu)
