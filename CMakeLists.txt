cmake_minimum_required(VERSION 3.10)
project(MyDL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory so headers can be found
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files (all .cpp files in src/)
file(GLOB SOURCES "src/*.cpp")

# Create executable
add_executable(MyDLApp ${SOURCES})
