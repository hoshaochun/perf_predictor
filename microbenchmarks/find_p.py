from compute_perf import benchmark_matmul
import numpy as np
from scipy.optimize import curve_fit
import os
import json

# 1. Define the objective function (The Lp Norm)
def smoothed_roofline(X, p):
    """
    X: A tuple or 2D array containing (T_comp, T_mem)
    p: The model parameter we want to find
    """
    t_comp, t_mem = X
    
    # Using np.power avoids domain errors with fractional exponents on arrays
    return np.power(np.power(t_comp, p) + np.power(t_mem, p), 1.0 / p)

def predict_compute_time(dim, compute_perf):
    M, N, K = dim
    total_flops = 2 * M * N * K
    bf16_flops = compute_perf["bf16"] * 1e12

    return total_flops / bf16_flops


def predict_memory_time(dim, memory_perf):
    M, N, K = dim
    total_bytes = 2 * (M*K + K*N + M*N)

    def get_achievable_bandwidth(data_size_bytes, memory_perf):
        data_size_mb = data_size_bytes / (1024 ** 2)
        
        bw_data = {}
        # convert the data size key to int
        for data_size in memory_perf:
            bw_data[int(data_size)] = memory_perf[data_size]
        
        # Extract X (MB) and Y (GB/s) points, sorted by X
        points = sorted(bw_data.items())

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
    
    mem_bandwidth = get_achievable_bandwidth(total_bytes, memory_perf)

    return total_bytes / mem_bandwidth


# get microbenchmark results
compute_file = "compute.json"
memory_file = "memory.json"

compute_perf = {}
with open(compute_file) as f:
    compute_perf = json.load(f)

memory_perf = {}
with open(memory_file) as f:
    memory_perf = json.load(f)


# target test cases to fit
H, I = 8192, 8192
B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]

T_comp = []
T_mem = []
T_measured = []
for b in B:
    T_comp.append(predict_compute_time((b, I, H), compute_perf) * 1e6)
    T_mem.append(predict_memory_time((b, I, H), memory_perf) * 1e6)
    T_measured.append(benchmark_matmul(b, H, I) * 1e6)

# print(T_comp)
# print(T_mem)
# print(T_measured)

# 2. Input your empirical data (Example Data)
# Replace these arrays with the actual measurements from your microbenchmark
T_comp = np.array(T_comp)       # Time strictly for compute
T_mem = np.array(T_mem)           # Time strictly for memory traffic
T_measured = np.array(T_measured)   # Actual measured time

# Pack the independent variables together
X_data = (T_comp, T_mem)

# 3. Run the curve fit
# p0 is the initial guess. 2.0 is a good starting point for modern GPUs.
# bounds=(1.0, np.inf) ensures p cannot be less than 1 (which would violate the physics of the model).
optimal_params, covariance = curve_fit(
    smoothed_roofline, 
    X_data, 
    T_measured, 
    p0=2.0, 
    bounds=(1.0, np.inf)
)

# Extract the fitted model parameter
p_fitted = optimal_params[0]

print(f"Fitted model parameter (p): {p_fitted:.3f}")

# 4. Verify the fit by calculating the modeled times
T_modeled = smoothed_roofline(X_data, p_fitted)

print("\n--- Verification ---")
for i in range(len(T_measured)):
    abs_err = abs(T_measured[i] - T_modeled[i])
    rel_err = (abs_err / T_measured[i]) * 100.0
    print(f"Measured: {T_measured[i]:.2f} us | Modeled: {T_modeled[i]:.2f} us | Relative Error: {rel_err:.2f}%")

# write result file
result_file = "result.json"
results = {}
results["compute"] = compute_perf
results["memory"] = memory_perf
results["p"] = p_fitted

with open(result_file, "w") as f:
    json.dump(results, f, indent=4)