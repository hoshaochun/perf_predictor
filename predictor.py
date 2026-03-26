import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict
import json
import numpy as np

expert_latency_cache = {}

@dataclass
class OperationLatency:
    """Stores compute, memory, and the bound latency for a single operation."""
    c: float = 0.0
    m: float = 0.0
    bound: float = 0.0  # Equivalent to "c+m" in the original code

    def calculate_bound(self):
        """Calculates the bound (max of compute or memory)."""
        self.bound = max(self.c, self.m)

    def __add__(self, other: 'OperationLatency') -> 'OperationLatency':
        if not isinstance(other, OperationLatency):
            return NotImplemented
        return OperationLatency(
            c=self.c + other.c,
            m=self.m + other.m,
            bound=self.bound + other.bound
        )


def default_latency_dict() -> Dict[str, OperationLatency]:
    """Provides a default dictionary for operation latencies."""
    return {
        "qkv": OperationLatency(),
        "attn_score": OperationLatency(),
        "o": OperationLatency(),
        "up_proj": OperationLatency(),
        "down_proj": OperationLatency()
    }


def sum_latencies(latency_dict: Dict[str, OperationLatency]) -> OperationLatency:
    """Helper function to aggregate total compute, memory, and bound across a dictionary."""
    total = OperationLatency()
    for op in latency_dict.values():
        total += op
    return total

def add_latency_dicts(l1: Dict[str, OperationLatency], l2: Dict[str, OperationLatency]):
    """Adds l2 to l1"""
    for op in l1:
        l1[op] += l2[op]

@dataclass
class Latency:
    waiting: OperationLatency = field(default_factory=OperationLatency)
    prefill: Dict[str, OperationLatency] = field(default_factory=default_latency_dict)
    decode: Dict[str, OperationLatency] = field(default_factory=default_latency_dict)


@dataclass
class Request:
    start_time: float
    input_len: int
    output_len: int
    id: Any
    finish_time: float = 0
    kv_len: int = 0
    cur_input_len: int = 0
    latency: Latency = field(default_factory=Latency)


def num_tokens_in_batch(requests: List[Request]) -> int:
    return sum(r.cur_input_len for r in requests)

def get_achievable_bandwidth(data_size_bytes, gpu):
    """
    Estimates the achievable memory bandwidth for a given data size in bytes.
    
    :param data_size_bytes: Arbitrary data size in bytes
    :return: Estimated bandwidth in GB/s
    """
    gpu_model = gpu.name

    # load bandwidth data for better estimation
    with open("microbenchmarks/mem_bandwidth.json") as f:
        data = json.load(f)

    bw_data = {}
    # convert the data size key to int
    for gpu in data:
        gpu_bw_data = {}
        for data_size in data[gpu]:
            gpu_bw_data[int(data_size)] = data[gpu][data_size]
        bw_data[gpu] = gpu_bw_data

    if gpu_model not in bw_data:
        raise ValueError(f"Unknown GPU model: {gpu_model}")
        
    # Convert arbitrary bytes to Megabytes (1 MB = 1024^2 bytes)
    # Note: If your microbenchmark treats 1MB as 10^6 bytes, change this to 1_000_000
    data_size_mb = data_size_bytes / (1024 ** 2)
    
    # Extract X (MB) and Y (GB/s) points, sorted by X
    points = sorted(bw_data[gpu_model].items())

    x_mb = np.array([p[0] for p in points])
    y_bw = np.array([p[1] for p in points])
    
    # Use log2 space for the X-axis because the benchmark grows exponentially 
    # and bandwidth scales logarithmically with cache boundaries
    x_log = np.log2(x_mb)
    data_size_log = np.log2(data_size_mb)
    
    # Interpolate (numpy.interp automatically handles out-of-bounds by capping 
    # to the minimum or maximum Y values)
    estimated_bw = np.interp(data_size_log, x_log, y_bw)
    return estimated_bw * 1e9


def matmul_latency(n: int, h1: int, h2: int, gpu, precision: int = 2) -> Tuple[float, float]:
    """
    Calculates compute and memory latency for a matrix multiplication.
    Compute: 2 * N * h1 * h2 FLOPs
    Memory: Precision * (Inputs + Weights + Outputs) bytes
    """
    compute_time = (2 * n * h1 * h2) / gpu.bf16_flops

    total_bytes = precision * (n * h1 + h1 * h2 + n * h2)
    mem_bandwidth = get_achievable_bandwidth(total_bytes, gpu)
    memory_time = total_bytes / gpu.mem_bandwidth
    return compute_time, memory_time


def GGEMM_latency(shapes: List[Tuple[int, int, int]], gpu, act_precision: int = 2, weight_precision: int = 2) -> Tuple[float, float]:
    total_flops = 0
    total_bytes = 0
    for s in shapes:
        n, h1, h2 = s
        if n == 0:
            continue
        total_flops += 2 * n * h1 * h2
        total_bytes += act_precision * (n * h1 + n * h2) + weight_precision * (h1 * h2)

    compute_time = total_flops / gpu.bf16_flops

    mem_bandwidth = get_achievable_bandwidth(total_bytes, gpu)
    memory_time = total_bytes / mem_bandwidth

    return compute_time, memory_time


def attn_score_latency(requests: List[Request], h: int, g: int, max_window_len: int, gpu, precision: int = 2) -> Tuple[float, float]:
    total_flops = 0
    total_bytes = 0
    
    for r in requests:
        n_token = r.cur_input_len
        n_kv = min(r.cur_input_len + r.kv_len, max_window_len)

        total_flops += 4 * n_token * n_kv * h
        total_bytes += precision * (2 * n_kv * h // g + 2 * n_token * h)

    compute_time = total_flops / gpu.bf16_flops

    mem_bandwidth = get_achievable_bandwidth(total_bytes, gpu)
    memory_time = total_bytes / mem_bandwidth

    return compute_time, memory_time


def transformer_latency(requests: List[Request], layer: int, gpu, model) -> Dict[str, OperationLatency]:
    latency = default_latency_dict()
    n_token = num_tokens_in_batch(requests)

    # ---------------------------------------------------------
    # GQA Attention
    # ---------------------------------------------------------
    g = model.n_attention_heads // model.n_kv_heads

    # 1. Linear Projections (Q, K, V)
    c, m = matmul_latency(
        n_token, 
        model.hidden_size, 
        model.head_dim * (model.n_attention_heads + 2 * model.n_kv_heads), 
        gpu
    )
    latency["qkv"] = OperationLatency(c, m)

    # 2. Attention Core (QK^T and Score * V)
    c, m = attn_score_latency(
        requests, 
        model.head_dim * model.n_attention_heads, 
        g, 
        model.max_seq_len, 
        gpu
    )
    latency["attn_score"] = OperationLatency(c, m)

    # 3. Output Projection
    c, m = matmul_latency(
        n_token, 
        model.head_dim * model.n_attention_heads, 
        model.hidden_size, 
        gpu
    )
    latency["o"] = OperationLatency(c, m)

    # ---------------------------------------------------------
    # Mixture of Experts (FFN)
    # ---------------------------------------------------------
    if n_token in expert_latency_cache:
        latency["up_proj"] = expert_latency_cache[n_token]["up_proj"]
        latency["down_proj"] = expert_latency_cache[n_token]["down_proj"]

    else:
        # For now, we assume uniform distribution
        w_bytes = 2
        a_bytes = 2
        if model.quantization == "mxfp4":
            w_bytes = 0.5

        expert_list = [i for i in range(model.n_experts)]
        selected_experts = []
        for i in range(n_token):
            selected = random.sample(expert_list, model.top_k)
            selected_experts.append(list(selected))

        token_counts = [0 for i in range(model.n_experts)]
        for i in range(n_token):
            for e in selected_experts[i]:
                token_counts[e] += 1

        shapes = []
        for i in range(model.n_experts):
            shapes.append((token_counts[i], model.hidden_size, model.moe_intermediate_size * 2))
        # print(token_counts)
        # 1. Up & Gate Projection
        # h -> 2 * inter
        # up_proj_time = matmul_latency(n_token, hidden_dim, intermediate_dim * 2, gpu)
        c, m = GGEMM_latency(shapes, gpu, a_bytes, w_bytes)
        latency["up_proj"] = OperationLatency(c, m)
        # if layer == 0:
        #     print(n_token, max(c, m))
        shapes = []
        for i in range(model.n_experts):
            shapes.append((token_counts[i], model.moe_intermediate_size, model.hidden_size))

        # 2. Down Projection
        # inter -> h
        # down_proj_time = matmul_latency(n_token, intermediate_dim, hidden_dim, gpu)
        c, m = GGEMM_latency(shapes, gpu, a_bytes, w_bytes)
        latency["down_proj"] = OperationLatency(c, m)

        expert_latency_cache[n_token] = {"up_proj": latency["up_proj"], "down_proj": latency["down_proj"]}

    # Calculate final bounds (max of compute/memory) per layer
    for op in latency.values():
        op.calculate_bound()
        
    return latency


def estimate_forward_latency(requests: List[Request], gpu, model) -> Dict[str, OperationLatency]:
    total_latency = default_latency_dict()
    
    for l in range(model.n_layers):
        layer_latency = transformer_latency(requests, l, gpu, model)
        # Add the current layer's latencies to the running totals
        add_latency_dicts(total_latency, layer_latency)

    return total_latency