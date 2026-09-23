cmake_minimum_required(VERSION 3.20)
project(indus_opt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter)
endif()

# Core include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Core linear algebra source files
set(LINALG_SOURCES
    src/linalg/sparse.cpp
    src/linalg/lu.cpp
    src/linalg/ldl.cpp
    src/linalg/scaling.cpp
)

# Simplex engine sources
set(SIMPLEX_SOURCES
    src/solvers/simplex/dual_simplex.cpp
    src/solvers/simplex/primal_simplex.cpp
)

# Library target
add_library(indus_core STATIC ${LINALG_SOURCES} ${SIMPLEX_SOURCES})
target_include_directories(indus_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Test executables
enable_testing()

add_executable(test_linalg tests/test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE indus_core)
add_test(NAME test_linalg COMMAND test_linalg)

add_executable(test_simplex tests/test_simplex.cpp)
target_link_libraries(test_simplex PRIVATE indus_core)
add_test(NAME test_simplex COMMAND test_simplex)
