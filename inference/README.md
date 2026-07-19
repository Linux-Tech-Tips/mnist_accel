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

TBW
