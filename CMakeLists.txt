cmake_minimum_required(VERSION 3.18)
project(quant_ml_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for VS Code / IDE IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Locate active Python interpreter in environment
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Retrieve LibTorch path directly from active Python instance
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

# Optimization flags for modern x86 CPU architectures
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

add_executable(quant_engine src/main.cpp)
target_link_libraries(quant_engine PRIVATE "${TORCH_LIBRARIES}")