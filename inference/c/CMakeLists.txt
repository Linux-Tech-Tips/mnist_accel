cmake_minimum_required(VERSION 3.23)

project(MNIST_c_inference)

# Compile-time bool options
option(USE_NEON "shorthand to use NEON matmul implementation (overridden by MATMUL_IMPL if specified)")

# Setting variables based on options
set(MATMUL_IMPL_DEFAULT "matmul_base.c")
if(USE_NEON)
    set(MATMUL_IMPL_DEFAULT "matmul_neon.c")
endif()

# User-configurable CACHE variables
set(MATMUL_IMPL "${MATMUL_IMPL_DEFAULT}" CACHE STRING "imatmul.h implementation source file")

# Compiler options
add_compile_options(-g -O2)
if(CMAKE_SYSTEM_PROCESSOR EQUAL "aarch64")
    add_compile_options(-march=armv8.2-a)
endif()

# Executable
add_executable(inference.elf)
target_sources(inference.elf
    PRIVATE
	src/main.c
	src/network.c
	"src/${MATMUL_IMPL}"
)
