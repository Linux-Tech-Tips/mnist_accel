# MNIST Acceleration using Arm SIMD

This project consists of a feedforward neural network trained using PyTorch on the MNIST dataset,
combined with a C inference acceleration layer written using Arm64 NEON SIMD instructions and executed on a Raspberry Pi.

The primary objective is to learn more about both FNNs and PyTorch as well as SIMD, low level inference acceleration and CPU profiling.

## Training

The training was accomplished using PyTorch on the MNIST dataset.

After training, the PyTorch code holding the FNN in RAM exported the sufficient network's layers into a custom data format,
keeping the layer weights and biases easy to load in both Python and C.

For more details, see [the training directory README](training/README.md).

## Inference Implementation

The inference layer was implemented in 2 parts, firstly a Numpy reference implementation, then a C implementation with a variable matrix-vector multiplication kernel.

The C kernel, declared in an interface header `imatmul.h`, was implemented in 2 ways, firstly a simple, nested-for-loop universal C implementation,
and secondly using ACLE-provided NEON SIMD intrinsics, with an optimised memory pre-packing format included.

For more details, see [the inference directory README](inference/README.md).

## Performance Measurement Results

As is described in the inference directory README, performance measurement across Python and the 2 C implementations was achieved by capturing
current time before and after each network inference, and accumulated across multiple input images and repeats.
The bench mode of each program then reports an average (mean) time per inference in milliseconds, rounded to three significant figures.

The platform used for testing was a Raspberry Pi 5 with a headless installation of Raspberry Pi OS (Debian-based Linux, Arm64), accessed over SSH.
The performance test itself used 10,000 in-program repetitions over 5 program invocations, giving a total of 50,000 inferences.
In addition, each of the 5 invocations measured model load time, which was also averaged.

The results are enumerated in the following table:

| Statistic               | Python [ms] | C Base [ms] | C SIMD [ms] |
| ----------------------- | ----------: | ----------: | ----------: |
| Average inference time  |       0.254 |       0.508 |      0.0849 |
| Average model load time |        46.5 |       0.668 |        1.92 |

For a helpful visual reference, the following graphs generated using Python's MatPlotLib show the same figures:

![MatPlotLib graphs showing time averages](.assets/time.png)

It can be observed that the slowest implementation is the basic nested for loop C variant.
Even though the code itself was compiled with GCC's `-O2` optimization level (same as the SIMD variant), since the code doesn't contain any direct optimisations,
it runs the slowest, though with a still relatively fast about half a millisecond.

Even though the NumPy implementation is written in Python, it runs comparatively very fast thanks to NumPy's various optimisations.
As per the [official NumPy documentation](https://numpy.org/doc/stable/reference/simd/index.html), the library delegates its computationally heavy operations
to C code under the hood, making efficient use of multiplatform SIMD under the hood based on CPU architecture.

The increased execution time over the last implementation comes down most likely to Python orchestration and extra function call overhead,
as well as the code being more generally usable over the last implementation, which features code built specifically for this project's use case.

The fastest implementation by a large margin (nearly 3 times faster than the Python variant) is the specialised C SIMD version.
This comes down to the implementation, efficient use of memory, cache, CPU SIMD registers and instructions, described at a higher level in the inference directory README.

### SIMD Implementation Impact

To start, it can be noted that unlike the base C implementation, this variant actually makes use of the provided `imatmul_kernel_pack` function, pre-packing weights
into memory at model load time, which also explains the slightly higher average model load time.

The purpose of said pre-packing is to introduce a memory layout allowing sequential access to weights at inference time, which is more cache-efficient.
Because of cache spacial locality, the assumption that data in close proximity is likely to be accessed at similar times, a call to load data from RAM fetches
an entire contiguous cache line, typically more values than directly necessary.
This means that when the program next loads data into registers, rather than from main memory, it can be loaded from CPU cache directly, which is a much lower-latency operation.

The kernel itself makes efficient use of specialised SIMD registers and instructions, offered by the CPU's advanced SIMD extension known as Arm Neon.
Using SIMD, the kernel loads a column segment each iteration, multiplies it by the corresponding input activation, and accumulates into output registers,
using the VMLA instruction (Vector Multiply Accumulate).
The output registers are initially populated with the weights, which eliminates the need for both register zero-initialisation and an additional vector add instruction after.

Mathematically, the specific approach to matrix-vector multiplication can be represented as

```math
\mathbf{o_k} = \mathbf{B_k} + \sum_{j=0}^{N} i_j \mathbf{W_kj}
```

given input buffer $\mathbf{i}$ with the j-th element $i_j$, output buffer $\mathbf{o}$ with a k-long segment $\mathbf{o_k}$,
the bias buffer $\mathbf{B}$ with a k-long segment $\mathbf{B_k}$, as well as $N$ the width of the weight matrix $\mathbf{W}$,
with $\mathbf{W_kj}$ representing a k-long column segment of the j-th column of $\mathbf{W}$.
Iterated in steps of $k$ over the weight matrix height, this computes the layer's full output buffer.

By taking a column segment at a time and iterating horizontally over the weight matrix in a larger row (height is the size of the column segment),
we can parallelise over each output activation, calculating multiple full outputs each outer loop iteration, and loading each weight only once.
This avoids the need for a final sum across all SIMD register elements, as well as requiring less memory writes, since we only write to each element of the output buffer once.

This does come with the drawback of needing to read the entire input buffer multiple times, however, this was deemed an insignificant cost in comparison to where the approach provides uplift.

Additionally, since Arm Neon provides up to 32 64-bit or 16 128-bit vector registers, for further parallelisation, to reduce loop iteration count and for better instruction-level parallelism,
each SIMD step in the main loop is unrolled to use 4 registers at a time (4 input and 4 output), each 128-bit holding 4 32-bit IEEE754 floats.
This allows us to process 16 weights at a time.

The ReLU activation function is optimised as well, reading over the output buffer in blocks of 16,
and implementing the rectified linear unit function using a VABS (Vector Absolute Value), VADD (Vector Addition) and VMUL (Vector-scalar Multiplication).
The implementation follows the formula

```math
\mathbf{b} = \mathrm{ReLU}(\mathbf{a}) = \frac{|\mathbf{a}| + \mathbf{a}}{2}
```

Given input vector (layer output buffer) $\mathbf{a}$ and final activated layer output vector $\mathbf{b}$.


### CPU Profiling Measurements

To properly understand where the SIMD implementation gains the most and why over the base, nested for loop variant, we can profile CPU statistics
by making use of the CPU's built-in PMU (Performance Monitoring Unit) to set up and read hardware counters.

In the C project, optionally provided is a module consisting of the files `perf_inst.h` and `perf_inst.c`, providing CPU profiling function instrumentation.
Using GCC's default `-finstrument-functions` option and by implementing the expected `__cyg_profile_func_enter` and `__cyg_profile_func_exit` functions,
we can execute code each time a function is entered and each time it completes.

Using the Linux system call `perf_event_open` to open a file descriptor pointing to the PMU's counters along with the `read` function, we can gather performance statistics.
More details on how the feature is implemented are in the inference directory README.

Through the profiling instrumentation, we read 6 statistics, each averaged (mean) per matmul kernel function invocation:
**CPU cycles**, **retired (completed) instructions**, **branch instructions**, **branch misses**, **cache references**, **cache misses**.

The results are averaged (mean, rounded to nearest integer) over one external program invocation and a total of 100,000 inferences (10,000 repetitions on 10 images),
which is a total of 300,000 function invocations since the 3-layer MNIST network runs the matmul kernel three times per inference.

The numbers are provided in the following table:

| Measurement          | C base count | C SIMD count |
| -------------------- | -----------: | -----------: |
| CPU cycles           |      399,306 |       66,526 |
| retired instructions |    1,250,723 |      174,246 |
| branch instructions  |      178,906 |       12,125 |
| branch misses        |          267 |           24 |
| cache references     |      536,201 |       59,155 |
| cache misses         |          696 |        1,031 |

For convenience, the following tables showcase the same figures, the branch and cache-related statistics in logarithmic scale:

![graphs showing numeric CPU profiling averages](.assets/cpu_measurements.png)

From the values in the table, we can compute a few synthetic statistics, notably IPC (instructions per cycle) and branch misprediction/cache miss percentages:

| Statistic              |  C base |  C SIMD |
| ---------------------- | ------: | ------: |
| IPC                    |    3.13 |    2.62 |
| branch misprediction % |   0.15% |   0.20% |
| cache miss %           |   0.13% |   1.74% |

In addition, we can produce other useful synthetic statistics by comparing the measurements of the base implementation directly against SIMD (dividing the base numeric value with the SIMD one):

| Statistic            | base/SIMD |
| -------------------- | --------: |
| CPU cycles           |      6.00 |
| retired instructions |      7.18 |
| branch instructions  |      14.8 |
| branch misses        |      11.1 |
| cache references     |      9.06 |
| cache misses         |     0.675 |

From the data we can observe a few patterns that hint at what's going on in the kernels that makes the SIMD implementation so much faster.

One very notable numeric difference is the much larger CPU cycle count and total instructions executed count of the base implementation.
This is almost certainly the primary difference, as by using less cycles and less instructions, the SIMD implementation takes less total time on the CPU leading to faster inference times.
Especially notable here is the last table, showing that the base implementation takes 6 times as many cycles with more than 7 times as many instructions.

The last table also points to the SIMD implementation using instructions more efficiently by branching less (the base implementation using nearly 15 times as many branch instructions),
mispredicting branches less, and referencing the cache less (due to less total instructions reading data thanks to the efficient use of wide SIMD registers).

An interesting side note is that through the IPC values greater than one, we can directly observe a superscalar CPU in action.
Thanks to approaches such as pipelining, the superscalar paradigm lets a CPU core execute multiple instructions at once by delegating
different parts of instructions (decode, arithmetic, memory/cache access etc.) to different execution units.
The matmul functions execute an average of between 2.5 and 3 instructions per CPU clock cycle, leading to greater efficiency.

From the IPC, we can estimate that the SIMD implementation is actually somewhat more memory-bound than the for loop counterpart, as a lower IPC value may indicate
longer time spent waiting on load/store operations to finish, while a higher IPC means more computation is likely done per clock cycle,
though in this case, the extra branch instructions introduced by the nested for loops may skew the figures somewhat.

The higher IPC and lower percentage of branch and cache misses points to the base implementation utilising memory and cache in a more optimal way in theory,
which is consistent with GCC's many built-in optimisations which tend to lead to compiled machine code running very efficiently.

However, while the SIMD implementation may be slightly less efficient in its memory accesses in theory, practically the way it accesses memory
and the much higher SIMD parallelisation it allows us to do, leads to that implementation being the clear winner here,
taking much less total CPU cycles, branching less and, as such, executing the matrix-vector multiplication operation significantly faster.

