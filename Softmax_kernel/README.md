# Row-wise Softmax with Triton

A small custom GPU implementation of row-wise softmax for two-dimensional
PyTorch tensors. The kernel uses Triton for the row reductions and a PyTorch
implementation as a correctness and performance reference.

This is useful as a compact learning and experimentation project. It is not
yet a general-purpose replacement for `torch.softmax`: read the input and
runtime limitations below before using it with tensors other than the
contiguous CUDA `float32` case exercised by the included test and benchmark.

## What it computes

For each row \(i\) and column \(j\), softmax is

\[
y_{i,j} = \frac{\exp(x_{i,j})}{\sum_k \exp(x_{i,k})}.
\]

The Triton kernel subtracts the maximum value in each row before taking
exponentials:

\[
y_{i,j} =
\frac{\exp(x_{i,j} - \max_k x_{i,k})}
{\sum_k \exp(x_{i,k} - \max_k x_{i,k})}.
\]

This max-subtraction is the usual numerically stable form for finite input
values. Each Triton program handles one or more rows. Columns are padded to
the next power of two, and masked loads/stores keep the padded elements out of
the result. The Python wrapper estimates a grid size from the active GPU's
properties and the compiled kernel's register/shared-memory metadata.

## Requirements and installation

- Python, PyTorch with CUDA support, a compatible NVIDIA GPU/driver, and
  Triton.
- Install a PyTorch build appropriate for your CUDA environment, following
  the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
- **Platform note:** the upstream `triton` package is intended for Linux.
  Windows users should install **`triton-windows`** instead of `triton`.
  The Python import remains `import triton`. Check that the chosen
  `triton-windows`, Python, PyTorch, CUDA, and GPU versions are compatible.

Linux, after installing the appropriate CUDA-enabled PyTorch build:

```bash
python -m pip install triton
```

Windows, after installing the appropriate CUDA-enabled PyTorch build:

```powershell
python -m pip install triton-windows
```

The kernel queries the active CUDA device at module import time. A working
CUDA-enabled PyTorch installation and an available CUDA device are therefore
needed even to import `softmax_kernel.py`; this module is not currently
CPU-only or importable as a no-GPU utility.

## Quick start

Run the included correctness check on the active CUDA device:

```bash
python softmax_kernel.py
```

Or call the function from Python:

```python
import torch

from softmax_kernel import softmax

x = torch.randn((1823, 781), device="cuda", dtype=torch.float32)
y = softmax(x)

reference = torch.softmax(x, dim=1)
torch.testing.assert_close(y, reference, atol=1e-3, rtol=1e-3)
```

`softmax(x)` returns a new tensor with the same shape and dtype as `x`. The
included test uses a `(1823, 781)` matrix and compares against
`torch.softmax(x, dim=1)` with absolute and relative tolerances of `1e-3`.

## Current input contract and limitations

The public wrapper checks that the input is two-dimensional, but it does not
validate all of the following conditions. Use the kernel only when they hold:

- **CUDA device:** the implementation is built around the active CUDA device.
  Put the input on that device before calling `softmax`.
- **Layout:** the kernel honors the row stride but assumes columns are
  contiguous (`x.stride(1) == 1`). Standard row-major contiguous matrices meet
  this condition. Transposed tensors and column-strided views are not
  supported correctly because the kernel does not use the column stride.
- **Dtype:** the included correctness test and benchmark create `float32`
  inputs. Other dtypes are not validated by this project.
- **Shapes and special values:** empty dimensions, very large row widths,
  rows containing infinities, and NaN behavior are not covered by the test.
  In particular, a row consisting entirely of negative infinity produces
  undefined `-inf - (-inf)` arithmetic, as it does with the usual max-shift
  formulation.
- **Device selection:** `DEVICE` is selected once when the module is imported.
  Switching the active CUDA device afterward is not a supported workflow.
  The `device` parameter on `test(...)` controls where its random input is
  created, so it must refer to the same active CUDA device used by the kernel.

The launch-size heuristic inspects compiled-kernel metadata and calls the
underscored `kernel._init_handles()` method. These are runtime/compiler
implementation details rather than a stable public Triton API, so behavior
may need adjustment when upgrading Triton or using a different GPU.

## API

### `softmax(x)`

Computes softmax along dimension 1 of a 2D tensor.

```python
y = softmax(x)
```

The current implementation accepts a two-dimensional CUDA tensor with
contiguous columns, as described above. The output is allocated with
`torch.empty_like(x)`.

### `simple_softmax(x)`

A straightforward PyTorch implementation of the same row-wise operation,
included as a readable reference. It is not the implementation used by the
benchmark's PyTorch provider; the benchmark compares against
`torch.softmax`.

### `test(size, a_tol=1e-3, r_tol=1e-3, device=DEVICE)`

Creates a random input, runs the Triton implementation, compares it with
`torch.softmax`, and prints `Test Passed` if the comparison succeeds. `size`
must be a two-element tuple. `device` selects the device used to create the
input; pass the active CUDA device the kernel was initialized for.

## Benchmark

Run the benchmark with:

```bash
python softmax_kernel.py --benchmark
```

It compares the custom Triton kernel with `torch.softmax` using `float32`
inputs and a fixed row count of `M = 4096`. The column count ranges from 256
through 12,672 in steps of 128. Results are reported as estimated effective
bandwidth in GB/s, calculated as two tensor-sized transfers (one read and one
write) divided by the measured time.

The benchmark is intended for local comparisons, not as a universal
performance claim. Results depend on the GPU, driver, CUDA/PyTorch/Triton
versions, clocks, and other workload conditions. No performance numbers are
guaranteed by this README.

## Correctness and project status

The row reduction and max-subtraction implement the standard stable softmax
algorithm, and the included check compares it with PyTorch. The source
currently exercises only a CUDA `float32` case. A successful check on one GPU
does not establish correctness for unsupported strides, other dtypes, empty
shapes, or every Triton/GPU version.

In short: this is a sound starting point for learning and benchmarking a
row-wise Triton softmax, but it should be treated as a constrained prototype
until input layout/device checks, device handling, and runtime compatibility
are made explicit or generalized.
