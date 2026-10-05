cmake_minimum_required(VERSION 3.18)
project(quant_ml_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)

# Find Python interpreter in the current active environment
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Execute python snippet to fetch PyTorch's CMake path
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_ERR
    RESULT_VARIABLE TORCH_RET
)

if(NOT TORCH_RET EQUAL 0)
    message(FATAL_ERROR "\n[-] PyTorch import failed using interpreter: ${Python3_EXECUTABLE}\n"
                        "[-] Please activate your virtual environment before running CMake:\n"
                        "    source ~/quant-ml-engine/venv/bin/activate\n")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

# Without a build type CMake compiles with no optimisation at all
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto")

# Trading engine: features + simulated quantum feature map + GRU + QUBO allocation
add_executable(quant_engine src/main.cpp)
target_link_libraries(quant_engine PRIVATE "${TORCH_LIBRARIES}")

# Offline model evaluation on a data split (accuracy, confusion matrix, latency)
add_executable(model_runner src/model_runner.cpp)
target_link_libraries(model_runner PRIVATE "${TORCH_LIBRARIES}")

# Unit tests: quantum simulator, C++/Python feature parity, AR estimator, QUBO/SA,
# ring buffer, portfolio accounting. No libtorch needed.
find_package(Threads REQUIRED)
add_executable(engine_tests tests/engine_tests.cpp)
target_link_libraries(engine_tests PRIVATE Threads::Threads)
enable_testing()
add_test(NAME engine_tests COMMAND engine_tests ${CMAKE_SOURCE_DIR}/tests/fixtures)