import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

def simple_softmax(x):
    max = x.max(dim = 1)[0]
    x = x - max[:, None]
    x = torch.exp(x)
    y = x.sum(dim = 1)
    x = x / y[:, None]
    return x


@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,                     
    BLOCK_SIZE: tl.constexpr,           
    num_stages: tl.constexpr,
): 
    
    row_start = tl.program_id(0) 
    row_step = tl.num_programs(0) 
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols

        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 


        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
N_SM = properties["multiprocessor_count"]
N_REGISTERS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def softmax(x):
    assert x.ndim == 2, "Only Matrices Are Allowed."
    n_rows, n_columns = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_columns)
    n_warps = 4
    if BLOCK_SIZE >= 2048:
        n_warps = 8
    if BLOCK_SIZE >= 4096:
        n_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM >= 200_000 else 2
    
    y = torch.empty_like(x)

    kernel = _softmax_kernel.warmup(x, y,
                                    x.stride(0), y.stride(0),
                                    n_rows, n_columns,
                                    BLOCK_SIZE= BLOCK_SIZE,
                                    num_stages = num_stages,
                                    num_warps = n_warps,
                                    grid = (1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_per_program = kernel.metadata.shared

    register_occupancy = N_REGISTERS // (n_warps * WARP_SIZE * n_regs)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_per_program
    min_bound = min(sram_occupancy, register_occupancy)  # Programs per SM
    n_programs = min(n_rows, N_SM * min_bound)
    grid = (n_programs, 1, 1)


    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_columns,
        BLOCK_SIZE,
        num_stages
    )

    return y


def test(size: tuple, a_tol = 1e-3, r_tol = 1e-3, device = DEVICE):
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2, 'size is incompatable'
    x = torch.randn(size[0], size[1], device = device)
    result1 = softmax(x)
    result2 = torch.softmax(x, dim = 1)
    torch.testing.assert_close(result1, result2, atol= a_tol, rtol= r_tol)
    print("Test Passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096}
    ))

def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)

if __name__ == "__main__":
    test(size=(1823, 781))

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)