cmake_minimum_required(VERSION 3.14)
project(matmul_free_llm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling multi-threading support")
else()
    message(STATUS "OpenMP not found, compiling single-threaded")
endif()

# Find all source files recursively
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Define target executable
add_executable(matmul_free_llm ${SOURCES})

# Include directory
target_include_directories(matmul_free_llm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul_free_llm PRIVATE OpenMP::OpenMP_CXX)
endif()
