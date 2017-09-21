cmake_minimum_required(VERSION 3.5)
project(chapter1)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hello_opencv hello_opencv.cc)
target_link_libraries(hello_opencv ${OpenCV_LIBS})
