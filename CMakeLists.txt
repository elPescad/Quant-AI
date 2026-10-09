cmake_minimum_required(VERSION 3.18)
project(quant_ml_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Without a build type CMake compiles with no optimisation at all
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# native = tuned for this machine. Binaries that run elsewhere (Docker images, cloud VMs)
# need a portable target: x86-64-v3 = AVX2 + FMA, available on every current x86 cloud CPU.
set(QUANT_ARCH "native" CACHE STRING "Value for -march")
option(QUANT_STATIC "Link fully static binaries (no shared libraries at runtime)" OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=${QUANT_ARCH} -fno-math-errno -fno-trapping-math -flto=auto")

include_directories(src)
find_package(Threads REQUIRED)

# Trading engine: features + simulated quantum feature map + native GRU + allocation
add_executable(quant_engine src/main.cpp)
# Offline model evaluation on a data split (accuracy, confusion matrix, latency)
add_executable(model_runner src/model_runner.cpp)
# Unit tests: quantum simulator, C++/Python feature and GRU parity, AR estimator,
# QUBO/SA, ring buffer, portfolio accounting
add_executable(engine_tests tests/engine_tests.cpp)

foreach(target quant_engine model_runner engine_tests)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    if(QUANT_STATIC)
        target_link_options(${target} PRIVATE -static)
    endif()
endforeach()

enable_testing()
add_test(NAME engine_tests COMMAND engine_tests ${CMAKE_SOURCE_DIR}/tests/fixtures)
