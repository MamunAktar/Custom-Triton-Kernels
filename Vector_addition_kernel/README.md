# Vector Addition with Triton

A small GPU vector-addition example implemented with [Triton](https://triton-lang.org/) and compared with PyTorch's built-in addition.

Given two input vectors, the kernel computes:

```text
z[i] = x[i] + y[i]
```

The project includes a correctness check and a benchmark that compares the custom kernel with `x + y`.

## How the kernel works

The Triton kernel assigns each program a block of `BLOCK_SIZE` elements:

1. `tl.program_id(0)` identifies the program's block.
2. The program builds element offsets for that block.
3. A mask prevents loads and stores beyond the end of the vectors, including in the final partial block.
4. It loads values from both inputs, adds them, and stores the results.

The wrapper launches enough programs to cover the vector and currently sets `BLOCK_SIZE` to `1024`.

## Requirements

- Python
- PyTorch installed with CUDA support
- Triton
- A CUDA-capable NVIDIA GPU and a compatible CUDA/PyTorch/Triton environment

Install PyTorch using the instructions for your CUDA environment at [pytorch.org](https://pytorch.org/get-started/locally/), then install Triton if it is not already available:

```bash
pip install triton
```

## Run

Run these commands from this directory.

Run the correctness check (the default script action):

```bash
python vector_addition_kernel.py
```

The test compares the custom kernel's result with PyTorch using `torch.testing.assert_close` and prints `TEST PASSED` if the values match.

Run the benchmark:

```bash
python vector_addition_kernel.py --benchmark
```

The benchmark tests vector sizes from `2**12` through `2**27` elements. Triton's benchmark runner writes its result artifacts to the current directory.

## Use the custom operation

```python
import torch

from vector_addition_kernel import add

device = torch.device("cuda")
x = torch.rand(1_000_000, device=device)
y = torch.rand(1_000_000, device=device)
z = add(x, y)
```

`add` allocates and returns a new output tensor; it does not modify either input. The current implementation is intended for same-shaped, contiguous input tensors on the CUDA device selected when the module is imported. The module selects the current CUDA device at import time, so a CUDA device must be available before importing it.

## Benchmark results

The checked-in chart and data are from the benchmark artifacts in this directory:

- [Benchmark chart](./CUSTOM%20KERNEL%20VS%20TORCH%20KERNEL.png)
- [Benchmark data (CSV)](./CUSTOM%20KERNEL%20VS%20TORCH%20KERNEL.csv)

The chart shows the custom kernel and PyTorch reaching roughly **124 GB/s** at the largest tested sizes in this run. The plotted bandwidth is calculated as:

```text
3 × number_of_elements × bytes_per_element / elapsed_time
```

This counts two input reads and one output write. Small inputs are more affected by launch and timing overhead; larger inputs are increasingly limited by memory bandwidth. These measurements are specific to the hardware, drivers, and software versions used to produce the artifacts. Those environment details are not recorded here, so treat the results as an example comparison, not a general performance guarantee.
