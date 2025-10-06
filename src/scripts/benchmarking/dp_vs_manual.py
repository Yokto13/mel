import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 1000
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class _WrappedSearcher(nn.Module):
    def __init__(self, kb_embs, num_neighbors):
        super().__init__()
        self.register_buffer("kb_embs", kb_embs)
        self.num_neighbors: int = num_neighbors

    def forward(self, x):
        dot_product = F.linear(x, self.kb_embs)
        _, top_indices = dot_product.topk(self.num_neighbors)
        return top_indices


def benchmark(f, input_data, warmup_iters=20, profile_iters=100):
    for _ in range(warmup_iters):
        with torch.inference_mode():
            f(input_data)
        torch.cuda.synchronize()

    time_sum = 0.0
    for _ in range(profile_iters):
        start = time.time()
        with torch.inference_mode():
            f(input_data)
        torch.cuda.synchronize()
        end = time.time()
        time_sum += end - start
    return time_sum / profile_iters


num_cuda_devices = torch.cuda.device_count()
print(f"Number of available CUDA devices: {num_cuda_devices}")

cuda_devices = [torch.device(f"cuda:{i}") for i in range(num_cuda_devices)]
print(f"CUDA devices: {cuda_devices}")

kb = torch.randn(1000, 768)
data = torch.randn(32, 768).to(cuda_devices[0])

searcher_dp = _WrappedSearcher(kb, num_neighbors=10)
searcher_dp = nn.DataParallel(searcher_dp, device_ids=cuda_devices)
searcher_dp = torch.compile(searcher_dp)
searcher_dp.to("cuda")


class ManualSearcher:
    def __init__(self, kb, device_ids, num_neighbors):
        self.device_ids = device_ids
        self.searchers = []
        for i, device in enumerate(device_ids):
            searcher = _WrappedSearcher(kb, num_neighbors=num_neighbors).to(device)
            self.searchers.append(searcher)

    def find(self, x):
        # Split input data across available devices
        inputs = nn.parallel.scatter(x, self.device_ids)
        # Compute on each device
        outputs = [
            searcher(input_chunk.to(device))
            for searcher, input_chunk, device in zip(self.searchers, inputs, self.device_ids)
        ]
        gathered = nn.parallel.gather(outputs, self.device_ids[0])
        return gathered


searcher_manual = ManualSearcher(kb, device_ids=cuda_devices, num_neighbors=10)
searcher_manual.find = torch.compile(searcher_manual.find)

print("Comparing outputs...")
out_dp = searcher_dp(data)
out_manual = searcher_manual.find(data)
print("Outputs are equal:", torch.equal(out_dp, out_manual))
assert torch.equal(out_dp, out_manual)

kb = torch.randn(10000000, 128).to(torch.float16)
data = torch.randn(256, 128).to(cuda_devices[0], dtype=torch.float16)

print("Benchmarking DataParallel compiled searcher...")
searcher_dp = _WrappedSearcher(kb, num_neighbors=10)
searcher_dp = nn.DataParallel(searcher_dp, device_ids=cuda_devices)
searcher_dp = torch.compile(searcher_dp)
searcher_dp.to("cuda")
print(benchmark(searcher_dp, data))
del searcher_dp

print("Benchmarking DataParallel searcher...")
searcher_dp = _WrappedSearcher(kb, num_neighbors=10)
searcher_dp = nn.DataParallel(searcher_dp, device_ids=cuda_devices)
searcher_dp.to("cuda")
print(benchmark(searcher_dp, data))
del searcher_dp

print("Benchmarking Manual searcher...")
searcher_manual = ManualSearcher(kb, device_ids=cuda_devices, num_neighbors=10)
searcher_manual.find = torch.compile(searcher_manual.find)
print(benchmark(searcher_manual.find, data))
del searcher_manual

print("Benchmarking Normal searcher...")
normal_searcher = _WrappedSearcher(kb, num_neighbors=10).to("cuda")
print(benchmark(normal_searcher, data))
del normal_searcher

print("Benchmarking Compiled Normal searcher...")
normal_searcher_c = _WrappedSearcher(kb, num_neighbors=10).to("cuda")
normal_searcher_c = torch.compile(normal_searcher_c)
print(benchmark(normal_searcher_c, data))
del normal_searcher_c
