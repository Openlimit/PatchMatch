cmake_minimum_required(VERSION 3.10)
project(PatchMatch)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(PatchMatch main.cpp PatchMatch.h)
target_link_libraries(PatchMatch ${OpenCV_LIBS})