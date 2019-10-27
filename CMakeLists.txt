# Made by Pedro Ramoneda. 27/10/19


cmake_minimum_required(VERSION 3.13)
project( sdi )

file(GLOB_RECURSE srcs ./src/*)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( sdi main.cpp )
target_link_libraries( sdi ${OpenCV_LIBS} )