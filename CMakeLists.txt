cmake_minimum_required(VERSION 3.6)   # CMake version check
project(Quoridor)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++11 standard

set(wxWidgets_ROOT_DIR D:/Dokumente/wxWidgets-3.0.2)
find_package(wxWidgets  REQUIRED core base )
include(${wxWidgets_USE_FILE})
# and for each of your dependent executable/library targets:
set(EXECUTABLE_OUTPUT_PATH ../)

set(SOURCE_FILES Board.cpp
        Board.h
        Game.cpp
        Game.h
        quoridoraner.cpp
        play.h
        Player.cpp
        Player.h)            # Add main.cpp file of project root directory as source file
add_executable(Quoridoraner ${SOURCE_FILES})       # Add executable target with source files listed in SOURCE_FILES variable

target_link_libraries(Quoridoraner ${wxWidgets_LIBRARIES})
target_link_libraries(Quoridoraner -static-libgcc -static-libstdc++)
