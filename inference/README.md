# MLP FNN Inference Segment

This part of the project consists of NumPy Python code, basic C code and Arm64-optimised C code capable of pre-trained multi-layer perceptron model inference.

## Example Data

In the directory `./data/` are sample data provided for convenience, notably `default.pt`, a PyTorch model trained to approx. 98.4% accuracy,
and 10 sample PNG images of digits 0-9 that can be used to run the model.

The `./data/` directory also contains a simplified file `default.dat`, which is a custom export of the PyTorch tensors in a simplified data file.
The file is created using the `../training/dump_bytes.py` script, and the format is documented in the `../training/` directory.
For convenience, both the PyTorch export file and the simplified data dump file (used for ease of loading here) are provided in the `./data/` directory.

## NumPy Code

The Python implementation is all in the `./python/numpy_inference.py` file, using the libraries specified in `./python/requirements.txt`,
which are NumPy for the actual matrix-vector multiplication and Pillow for PNG image loading.

The code implements a Network dataclass, consisting of a list of NetworkLayer dataclass instances.

Each NetworkLayer holds weights, bias and an activation function for the given layer,
checks weight/bias shape constraints using a `__post_init__` function, and has a `run` function to execute layer calculations on an input array.

The code executing the matrix-vector multiplication is incredibly simple, consisting of a call to `np.matmul(weights, input)`
followed by a numpy vector add operator (`+`) used to add the layer's bias.

The main function implements `run` functionality to run a single inference and print the result,
as well as `bench` functionality, allowing multiple images and a number of repeats, and printing model load time
as well as total and average inference times.

## C Code

The C program, compiled using CMake, implements the same `run` and `bench` functionality in its main.
In essence, it's structured very similarly to the NumPy program, just more cleanly modular.

It consists of the following files:

```
./src/
  main.c
  network.h
  network.c
  imatmul.h
  matmul_base.c
  matmul_neon.h
  matmul_neon.c
```

The `main.c` file implements the run/bench logic as well as network and image loading orchestration.

The `network.h`/`network.c` files provide network layer and network structures, using the `network_layer_t` and `network_t` structs, respectively,
as well as functions to initialise and load a network from a file and to run a network from a given struct on given image data.

The module `imatmul.h` defines common function signatures for matrix multiplication, memory layout pre-packing and activation.
Any implementing file should include `imatmul.h` and implement all functions from it, in addition to any extra functions if any are needed.

### Base Implementation

The base implementation, in the file `matmul_base.c`, uses no extra memory pre-packing and very simple for loop-based matmul and activation.

The `imatmul_kernel_run` function containing the actual run code consists of 2 nested for loops, 
iterating over the weights matrix height, then weights matrix width, then multiplying the corresponding weight with the corresponding input,
as well as adding the corresponding bias each outer loop iteration.

The `imatmul_activation_relu` function does a simple for loop through the input buffer, writing the value or 0 based on what the value is.

### NEON Intrinsics Implementation

The Arm64 NEON Intrinsics implementation, in the files `matmul_neon.h`/`matmul_neon.c`, uses the Arm ACLE (Arm C Library Extensions) header,
which provides functions (intrinsics) allowing direct calls to Arm64 extended SIMD assembly using the NEON co-processor.

The `imatmul_kernel_pack` function pre-packs the weight and bias data into a format in memory such that loads are contiguous,
which leads to better cache utilisations and speeds up memory IO.
More details on the specific functionality are in the documentation comment in `matmul_neon.c`.

The `imatmul_kernel_run` function contains efficient Arm64 NEON SIMD code, using Arm ACLE intrinsics and relying on the packing format,
to multiply weights and the input buffer.
Each SIMD call is unrolled into `Q` separate calls (in this implementation that being 4),
to make better use of the available SIMD registers.
More details on the specific functionality and design are in the doc comment in `matmul_neon.c`.

The `imatmul_activation_relu` function does a strided SIMD loop using unrolled calls to NEON intrinsics,
implementing the ReLU functionality in a more parallelised way.
More details on the specific functionality and design are in the doc comment in `matmul_neon.c`.


## Performance Testing

Inference for both implementations was timed by in-language time gathering functions, with a start time taken before each network inference and end time taken after, the differences accumulated.
Taking the time around the network inference call inside the `bench` repeats loop makes sure the for loop logic doesn't interfere with measurements.

### CPU Profiling

To get a more accurate read on the differences between the base implementation and SIMD implementation, I implemented a simple CPU profiler that I used
together with GCC's built-in function instrumentation feature, allowing the gathering of statistics like CPU cycle count, total retired instructions, cache references and cache misses,
per function call.

This is implemented using GCC's `-finstrument-functions` option (as documented in the [official GCC online docs](https://gcc.gnu.org/onlinedocs/gcc-9.2.0/gcc/Instrumentation-Options.html)),
which lets us define additional functions `__cyg_profile_func_enter` and `__cyg_profile_func_exit` called before any instrumented function is entered and after any is exitted.

To gather the actual data, the profiler makes use of the kernel's PMU driver, interfacing directly with the CPU's Performance Monitoring Unit.
This is accomplished using the `perf_event_open` syscall (as documented in [the linux manual](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)),
called through the C `syscall` function, as `glibc` doesn't provide a built-in wrapper.

The profiling data is gathered into a struct of stats collections, differentiated by function pointer, accumulated using the deltas
between each function entry and exit, and the total number of invocations is collected to calculate per-call averages.

The source code can be found in 2 additional files, `src/perf_inst.h` and `src/perf_inst.c`.

Instrumentation for the `imatmul.h` function implementations can be enabled using the CMake flag `-DINSTRUMENT_MATMUL=ON` when configuring the build.
Running the instrumented executable will print per-function totals and averages for the `imatmul.h` functions after execution completes.

