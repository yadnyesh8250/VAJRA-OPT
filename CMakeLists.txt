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

# Engine, Presolve & IO sources
set(IO_SOURCES
    src/engine/options.cpp
    src/engine/model.cpp
    src/engine/verifier.cpp
    src/presolve/presolve_lite.cpp
    src/presolve/presolve.cpp
    src/io/mps_reader.cpp
    src/io/lp_reader.cpp
    src/io/writer.cpp
)

# PDHG and GPU hardware abstraction sources
set(PDHG_SOURCES
    src/solvers/pdhg/pdhg.cpp
    src/solvers/gpu/gpu_context.cpp
)

# Optional NVIDIA CUDA compilation
option(INDUS_ENABLE_CUDA "Enable NVIDIA CUDA GPU acceleration" OFF)

if(INDUS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_compile_definitions(INDUS_HAS_CUDA=1)
        list(APPEND PDHG_SOURCES src/solvers/gpu/pdhg_cuda.cu)
        message(STATUS "[INDUS-OPT] CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "[INDUS-OPT] INDUS_ENABLE_CUDA was requested, but no CUDA compiler was found on this system. Building with CPU fallback.")
        set(INDUS_ENABLE_CUDA OFF)
    endif()
endif()

# Git commit & dirty status detection
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "sovereign-v1.0")
else()
    execute_process(
        COMMAND git status --porcelain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DIRTY_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_DIRTY_STATUS)
        set(GIT_COMMIT_HASH "${GIT_COMMIT_HASH}-dirty")
    endif()
endif()
add_compile_definitions(INDUS_GIT_COMMIT="${GIT_COMMIT_HASH}")

# Library target
add_library(indus_core STATIC ${LINALG_SOURCES} ${SIMPLEX_SOURCES} ${IO_SOURCES} ${PDHG_SOURCES})
target_include_directories(indus_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_compile_definitions(INDUS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Test executables
enable_testing()

add_executable(test_linalg tests/test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE indus_core)
add_test(NAME test_linalg COMMAND test_linalg)
set_tests_properties(test_linalg PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_simplex tests/test_simplex.cpp)
target_link_libraries(test_simplex PRIVATE indus_core)
add_test(NAME test_simplex COMMAND test_simplex)
set_tests_properties(test_simplex PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_io tests/test_io.cpp)
target_link_libraries(test_io PRIVATE indus_core)
add_test(NAME test_io COMMAND test_io)
set_tests_properties(test_io PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_regression tests/test_regression.cpp)
target_link_libraries(test_regression PRIVATE indus_core)
add_test(NAME test_regression COMMAND test_regression)
set_tests_properties(test_regression PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_presolve tests/test_presolve.cpp)
target_link_libraries(test_presolve PRIVATE indus_core)
add_test(NAME test_presolve COMMAND test_presolve)
set_tests_properties(test_presolve PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_pdhg tests/test_pdhg.cpp)
target_link_libraries(test_pdhg PRIVATE indus_core)
add_test(NAME test_pdhg COMMAND test_pdhg)
set_tests_properties(test_pdhg PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# Standalone independent verifier executable
add_executable(indus_verifier apps/verifier_cli.cpp)
target_link_libraries(indus_verifier PRIVATE indus_core)

# Benchmark runner executable
add_executable(indus_benchmark apps/benchmark_runner.cpp)
target_link_libraries(indus_benchmark PRIVATE indus_core)
add_test(NAME test_benchmark COMMAND indus_benchmark --out-dir "${CMAKE_CURRENT_BINARY_DIR}/benchmark_artifacts" --csv "${CMAKE_CURRENT_BINARY_DIR}/benchmark_results.csv")
set_tests_properties(test_benchmark PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_test(NAME test_python_verifier COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/validator/independent_verifier.py" "${CMAKE_CURRENT_SOURCE_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps" "${CMAKE_CURRENT_BINARY_DIR}/benchmark_artifacts/afiro.sol")
    set_tests_properties(test_python_verifier PROPERTIES DEPENDS test_benchmark WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
