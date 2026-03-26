// measure the achievable memory bandwidth for different data size
// compile: nvcc mem_bandwidth.cu -o mem_bandwidth
// run: ./mem_bandwidth


#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

// Macro for standard CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- WRITE-ONLY KERNEL ---
__global__ void writeKernelFloat4(float4* __restrict__ d_out, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Constant value to write
    float4 val = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    
    for (size_t i = idx; i < num_elements; i += stride) {
        d_out[i] = val;
    }
}

// --- READ-ONLY KERNEL ---
__global__ void readKernelFloat4(const float4* __restrict__ d_in, float* __restrict__ dummy_out, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (size_t i = idx; i < num_elements; i += stride) {
        float4 val = d_in[i];
        // Accumulate to prevent compiler from optimizing the read away
        sum += val.x + val.y + val.z + val.w; 
    }
    
    // Prevent dead-code elimination. The condition will almost certainly 
    // be false, but the compiler doesn't know that at compile time.
    if (idx == 0 && sum == -1.0f) { 
        dummy_out[0] = sum;
    }
}

enum class BenchmarkType { READ, WRITE };

void runBenchmark(size_t chunk_size_bytes, int iterations, int deviceId, BenchmarkType type) {
    size_t elements_per_chunk = chunk_size_bytes / sizeof(float4);
    
    // --- MEMORY POOL SETUP ---
    // Create a pool large enough to easily evict the L2 cache (e.g., 512 MB minimum)
    // We cycle through this pool to ensure we are always hitting DRAM.
    size_t min_pool_size = 512ULL * 1024 * 1024; 
    size_t pool_size = std::max(min_pool_size, chunk_size_bytes * 4);
    size_t num_chunks = pool_size / chunk_size_bytes;
    size_t actual_pool_bytes = num_chunks * chunk_size_bytes;

    float4 *d_pool;
    float *d_dummy_out; // Used only for read kernel
    CUDA_CHECK(cudaMalloc(&d_pool, actual_pool_bytes));
    CUDA_CHECK(cudaMalloc(&d_dummy_out, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    int gridSize = numSMs * 32; 

    // WARM-UP (Run once on the first chunk)
    if (type == BenchmarkType::READ) {
        readKernelFloat4<<<gridSize, blockSize>>>(d_pool, d_dummy_out, elements_per_chunk);
    } else {
        writeKernelFloat4<<<gridSize, blockSize>>>(d_pool, elements_per_chunk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK ---
    // Record start event ONCE before the loop
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        // Cycle through the memory pool
        size_t chunk_idx = i % num_chunks;
        size_t offset_in_elements = chunk_idx * elements_per_chunk;

        if (type == BenchmarkType::READ) {
            readKernelFloat4<<<gridSize, blockSize>>>(d_pool + offset_in_elements, d_dummy_out, elements_per_chunk);
        } else {
            writeKernelFloat4<<<gridSize, blockSize>>>(d_pool + offset_in_elements, elements_per_chunk);
        }
    }

    // Record stop event ONCE after the loop and wait for all kernels to finish
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));

    // --- MATH ---
    double seconds = total_milliseconds / 1000.0;
    double total_bytes_transferred = 1.0 * chunk_size_bytes * iterations; 
    double bandwidth_GBps = (total_bytes_transferred / seconds) / 1e9;

    std::string type_str = (type == BenchmarkType::READ) ? "Read" : "Write";

    std::cout << std::left << std::setw(15) << (chunk_size_bytes / (1024 * 1024)) 
              << std::setw(15) << type_str
              << std::setw(20) << bandwidth_GBps << " GB/s" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFree(d_dummy_out));
}

int main() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    
    std::cout << "Testing on GPU: " << props.name << "\n";
    std::cout << "Memory Clock:   " << props.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Bus Width:      " << props.memoryBusWidth << " bits\n";
    std::cout << "L2 Cache Size:  " << props.l2CacheSize / (1024 * 1024) << " MB\n";
    
    double theoretical_bandwidth = 2.0 * (props.memoryClockRate * 1000.0) * (props.memoryBusWidth / 8.0) / 1e9;
    std::cout << "Theoretical Peak Bandwidth: " << theoretical_bandwidth << " GB/s\n";
    std::cout << std::string(55, '-') << "\n";
    
    std::cout << std::left << std::setw(15) << "Data Size (MB)" 
              << std::setw(15) << "Type"
              << std::setw(20) << "Achieved Bandwidth" << std::endl;
    std::cout << std::string(55, '-') << "\n";

    // Because we've removed the loop overhead, we can run more iterations very quickly
    int base_iterations = 200; 
    
    for (size_t size_MB = 1; size_MB <= 8192; size_MB *= 2) {
        size_t size_bytes = size_MB * 1024 * 1024;
        
        int iterations = base_iterations * (1024 / size_MB); 
        if (iterations > 2000) iterations = 2000;
        if (iterations < 50) iterations = 50;
        
        // Run both Read and Write benchmarks
        runBenchmark(size_bytes, iterations, deviceId, BenchmarkType::READ);
        // runBenchmark(size_bytes, iterations, deviceId, BenchmarkType::WRITE);
    }

    return 0;
}