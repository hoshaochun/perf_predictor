import torch
from cupti import cupti
import json

def benchmark_matmul(B, H, I):
    min_memory = int(1 * 10**9)
    mat_bytes = 2*(B*H + I*H + B*I)
    n_copies = max(10, min_memory // mat_bytes)
    n_warmups = 1000
    n_tests = 2000
    dtype = torch.bfloat16  # float16 is standard for LLM/DL performance
    device = torch.device("cuda")

    # 1. Setup - Define matrix dimensions
    # Shape: [B, S, H] x [H, I] -> [B, S, I]
    # Initialize tensors
    Xs = [torch.randn((B, H), device=device, dtype=dtype) for _ in range(n_copies)]
    Ws = [torch.randn((I, H), device=device, dtype=dtype) for _ in range(n_copies)]

    # 2. Warm-up
    for i in range(n_warmups):
        i = i % n_copies
        Y = torch.matmul(Xs[i], Ws[i].mT)

    torch.cuda.synchronize()  # Wait for warm-up to finish

    # 3. Setup CUPTI Callbacks
    # CUPTI needs a buffer to store the activity records. 
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0
        return buffer_size, max_num_records

    kernel_metrics = []
    
    # This callback is triggered when the buffer is full or flushed
    def func_buffer_completed(activities: list):
        for activity in activities:
            # We specifically want concurrent kernels, which represent the actual compute
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                duration_ns = activity.end - activity.start
                kernel_metrics.append({
                    "name": activity.name,
                    "duration_ns": duration_ns
                })

    # 4. Register and enable CUPTI
    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

    # 5. Execute the target PyTorch operation
    for i in range(n_tests):
        i = i % n_copies
        Y = torch.matmul(Xs[i], Ws[i].mT)
    
    # Ensure the GPU finishes the work before we flush CUPTI
    torch.cuda.synchronize()

    # 6. Flush buffers and disable collection
    cupti.activity_flush_all(1)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)

    # 7. Process and print results
    total_time_ns = 0
    # print(f"--- CUPTI PyTorch Matmul ({M}x{N}x{K}) Profiling ---")
    for idx, k in enumerate(kernel_metrics):
        # print(f"Kernel {idx + 1}: {k['name']}")
        # print(f"  Duration: {k['duration_ns'] / 1000.0:.2f} µs")
        total_time_ns += k['duration_ns']
        
    # print("-" * 40)
    # print(f"Total GPU Execution Time: {total_time_ns / 1000.0:.2f} µs")
    avg = total_time_ns / 1000**3 / n_tests

    return avg

M, N, K = 8192, 8192, 8192
latency = benchmark_matmul(M, K, N)

total_flops = 2 * M * N * K
achieved_flops = total_flops / latency

results = {}
results["bf16"] = achieved_flops / 1e12

with open("compute.json", "w") as file:
    json.dump(results, file, indent=4)

