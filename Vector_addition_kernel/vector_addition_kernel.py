import torch

import triton

import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit

def addkernel(x_add, y_add, z_add, n, BLOCK_SIZE:tl.constexpr):# x:input tensor, y:input tensor, z: output tensor, n = no. of elements
  program_id = tl.program_id(axis = 0)
  block_start = program_id * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n
  x = tl.load(x_add + offsets, mask = mask, other = None)
  y = tl.load(y_add + offsets, mask = mask, other = None)
  result = x + y
  tl.store(z_add + offsets, result, mask = mask)

def add(x:torch.tensor, y:torch.tensor):
  result = torch.empty_like(x)
  assert x.device == DEVICE and y.device == DEVICE and result.device == DEVICE, f"DEVICE:{DEVICE}, x_loc:{x.device}, y_loc:{y.device}, result_loc:{result.device}"
  n = result.numel()
  grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
  addkernel[grid](x, y, result, n, BLOCK_SIZE = 1024)
  return result

def test(size, a_tol = 1e-3, r_tol = 1e-3, device = DEVICE):
  torch.manual_seed = 0
  x = torch.rand(size, device = device)
  y = torch.rand(size, device = device)
  kernelvalue = add(x, y)
  torchvalue = x + y
  torch.testing.assert_close(kernelvalue, torchvalue, atol = a_tol, rtol = r_tol)
  print("TEST PASSED")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log= True,
        line_arg= "provider",
        line_vals=['kernel', 'torch'],
        line_names=['KERNEL', 'TORCH'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel= 'GB/S',
        plot_name='CUSTOM KERNEL VS TORCH KERNEL',
        args={},
    )
)

def benchmark(size, provider):
  x = torch.rand(size, device = DEVICE, dtype = torch.float32)
  y = torch.rand(size, device = DEVICE, dtype = torch.float32)

  quantiles = [.5, .05, .95]

  if provider == 'torch':
    half_ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles = quantiles)
  if provider == 'kernel':
    half_ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles = quantiles)

  gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9/(ms*1e-3)

  return gbps(half_ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__" :
  test(size = 98432)

  import sys
  if len(sys.argv) > 1:
    if sys.argv[1] == "--benchmark":
      benchmark.run(save_path = ".", print_data = False)