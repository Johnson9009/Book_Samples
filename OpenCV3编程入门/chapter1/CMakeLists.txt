cmake_minimum_required(VERSION 3.5)
project(chapter1)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hello_opencv hello_opencv.cc)
target_link_libraries(hello_opencv ${OpenCV_LIBS})

add_executable(erode erode.cc)
target_link_libraries(erode ${OpenCV_LIBS})

add_executable(blur blur.cc)
target_link_libraries(blur ${OpenCV_LIBS})

add_executable(canny canny.cc)
target_link_libraries(canny ${OpenCV_LIBS})

add_executable(video video.cc)
target_link_libraries(video ${OpenCV_LIBS})

add_executable(camera camera.cc)
target_link_libraries(camera ${OpenCV_LIBS})

add_executable(canny_camera canny_camera.cc)
target_link_libraries(canny_camera ${OpenCV_LIBS})
