cmake_minimum_required(VERSION 3.23)

project(MNIST_c_inference)

# Compile-time bool options
option(USE_NEON "shorthand to use NEON matmul implementation (overridden by MATMUL_IMPL if specified)")
option(INSTRUMENT_MATMUL "whether to add custom profiling instrumentation to the matmul functions")

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

if(INSTRUMENT_MATMUL)
    add_compile_options(-finstrument-functions -finstrument-functions-exclude-file-list=src/main,src/network,src/perf_inst,include/aarch64-linux-gnu)
    add_compile_definitions(INSTRUMENT_MATMUL)
    add_link_options(-rdynamic)
endif()

# Executable
add_executable(inference.elf)
target_sources(inference.elf
    PRIVATE
	src/main.c
	src/network.c
	"src/${MATMUL_IMPL}"
)
if(INSTRUMENT_MATMUL)
    target_sources(inference.elf PRIVATE src/perf_inst.c)
    target_link_libraries(inference.elf dl)
endif()
