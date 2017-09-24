# ake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(chapter2)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
# Link your application with OpenCV libraries

add_executable(cam_shift_demo cam_shift_demo.cc)
target_link_libraries(cam_shift_demo ${OpenCV_LIBS})

add_executable(lk_demo lk_demo.cc)
target_link_libraries(lk_demo ${OpenCV_LIBS})

add_executable(object_detec object_detec.cc)
target_link_libraries(object_detec ${OpenCV_LIBS})
