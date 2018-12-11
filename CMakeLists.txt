cmake_minimum_required(VERSION 3.7)   # CMake version check
project(Quoridor)
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard

set(SOURCE_FILES Board.cpp
        Board.h
        Game.cpp
        Game.h
        quoridoraner.cpp
        play.h
        Player.cpp
        Player.h)
add_executable(Quoridoraner ${SOURCE_FILES})       # Add executable target with source files listed in SOURCE_FILES variable

find_package(wxWidgets  REQUIRED core base )
set(wxWidgets_Configuration mswu)

include(${wxWidgets_USE_FILE})
#add_library(Quoridoraner ${wxWidets_LIBRARIES})
target_link_libraries(Quoridoraner ${wxWidgets_LIBRARIES})
#target_link_libraries(Quoridoraner -static-libgcc -static-libstdc++)
