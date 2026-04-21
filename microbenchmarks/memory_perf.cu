// measure the achievable memory bandwidth for different data size

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cupti.h>

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

// --- CUPTI GLOBALS & MACROS ---
#define CUPTI_CHECK(call) \
do { \
    CUptiResult _status = call; \
    if (_status != CUPTI_SUCCESS) { \
        const char *errstr; \
        cuptiGetResultString(_status, &errstr); \
        std::cerr << "CUPTI error: " << errstr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct CuptiMetrics {
    uint64_t total_kernel_time_ns;
    uint32_t kernel_count;
};

// Global to hold our accumulated time
CuptiMetrics g_metrics = {0, 0};

struct BenchmarkResult {
    size_t size_MB;
    std::string type;
    double bandwidth_GBps;
};

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



// Called by CUPTI when it needs a buffer to store activity records
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size = 16384; // 16 KB buffer
    *buffer = (uint8_t *)malloc(*size + 8); 
    *maxNumRecords = 0;
}

// Called by CUPTI when a buffer is full or flushed
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                // Check for concurrent kernel execution records
                if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || 
                    record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                    
                    // NOTE: The struct version (e.g., CUpti_ActivityKernel5, CUpti_ActivityKernel9) 
                    // depends on your CUDA Toolkit version. 
                    // CUpti_ActivityKernel5 is widely compatible with CUDA 11+.
                    CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
                    
                    // Accumulate exact hardware execution time
                    g_metrics.total_kernel_time_ns += (kernel->end - kernel->start);
                    g_metrics.kernel_count++;
                }
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
        } while (1);
    }
    free(buffer);
}

enum class BenchmarkType { READ, WRITE };

BenchmarkResult runBenchmark(size_t chunk_size_bytes, int iterations, int deviceId, BenchmarkType type) {
    size_t elements_per_chunk = chunk_size_bytes / sizeof(float4);
    
    // --- MEMORY POOL SETUP ---
    size_t min_pool_size = 512ULL * 1024 * 1024; 
    size_t pool_size = std::max(min_pool_size, chunk_size_bytes * 4);
    size_t num_chunks = pool_size / chunk_size_bytes;
    size_t actual_pool_bytes = num_chunks * chunk_size_bytes;

    float4 *d_pool;
    float *d_dummy_out; 
    CUDA_CHECK(cudaMalloc(&d_pool, actual_pool_bytes));
    CUDA_CHECK(cudaMalloc(&d_dummy_out, sizeof(float)));

    int blockSize = 256;
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    int gridSize = numSMs * 32; 

    // WARM-UP (Run once on the first chunk - not profiled)
    if (type == BenchmarkType::READ) {
        readKernelFloat4<<<gridSize, blockSize>>>(d_pool, d_dummy_out, elements_per_chunk);
    } else {
        writeKernelFloat4<<<gridSize, blockSize>>>(d_pool, elements_per_chunk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- SETUP CUPTI FOR BENCHMARK ---
    // Reset global metrics for this run
    g_metrics.total_kernel_time_ns = 0;
    g_metrics.kernel_count = 0;

    // Register callbacks and enable profiling for kernels
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // --- BENCHMARK ---
    for (int i = 0; i < iterations; ++i) {
        size_t chunk_idx = i % num_chunks;
        size_t offset_in_elements = chunk_idx * elements_per_chunk;

        if (type == BenchmarkType::READ) {
            readKernelFloat4<<<gridSize, blockSize>>>(d_pool + offset_in_elements, d_dummy_out, elements_per_chunk);
        } else {
            writeKernelFloat4<<<gridSize, blockSize>>>(d_pool + offset_in_elements, elements_per_chunk);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // --- FLUSH & DISABLE CUPTI ---
    // Disabling and flushing forces CUPTI to push all remaining records to the bufferCompleted callback
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityFlushAll(0));

    // --- MATH ---
    // Convert accumulated nanoseconds to seconds
    double seconds = g_metrics.total_kernel_time_ns / 1e9;
    double total_bytes_transferred = 1.0 * chunk_size_bytes * iterations; 
    double bandwidth_GBps = (total_bytes_transferred / seconds) / 1e9;

    std::string type_str = (type == BenchmarkType::READ) ? "Read" : "Write";
    size_t size_MB = chunk_size_bytes / (1024 * 1024);

    // Cleanup
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFree(d_dummy_out));

    return { size_MB, type_str, bandwidth_GBps };
}

int main() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    
    // std::cout << "Testing on GPU: " << props.name << "\n";
    // std::cout << "Memory Clock:   " << props.memoryClockRate / 1000 << " MHz\n";
    // std::cout << "Bus Width:      " << props.memoryBusWidth << " bits\n";
    // std::cout << "L2 Cache Size:  " << props.l2CacheSize / (1024 * 1024) << " MB\n";
    
    // double theoretical_bandwidth = 2.0 * (props.memoryClockRate * 1000.0) * (props.memoryBusWidth / 8.0) / 1e9;
    // std::cout << "Theoretical Peak Bandwidth: " << theoretical_bandwidth << " GB/s\n";
    // std::cout << std::string(55, '-') << "\n";
    
    // std::cout << std::left << std::setw(15) << "Data Size (MB)" 
    //           << std::setw(15) << "Type"
    //           << std::setw(20) << "Achieved Bandwidth" << std::endl;
    // std::cout << std::string(55, '-') << "\n";

    // Because we've removed the loop overhead, we can run more iterations very quickly
    int base_iterations = 200; 
    std::vector<BenchmarkResult> results;

    for (size_t size_MB = 1; size_MB <= 8192; size_MB *= 2) {
        size_t size_bytes = size_MB * 1024 * 1024;
        
        int iterations = base_iterations * (1024 / size_MB); 
        if (iterations > 2000) iterations = 2000;
        if (iterations < 50) iterations = 50;
        
        // Run both Read and Write benchmarks
        results.push_back(runBenchmark(size_bytes, iterations, deviceId, BenchmarkType::READ));
        // runBenchmark(size_bytes, iterations, deviceId, BenchmarkType::WRITE);

    }

    // --- WRITE JSON TO FILE ---
    std::string filename = "memory.json";
    std::ofstream json_file(filename);
    
    if (json_file.is_open()) {
        json_file << "{\n";
        
        for (size_t i = 0; i < results.size(); ++i) {     
            json_file << "    \"" << results[i].size_MB << "\": " << results[i].bandwidth_GBps;
            
            // Add a comma after the line if it's not the last element
            if (i < results.size() - 1) {
                json_file << ",";
            }
            json_file << "\n";
        }
        
        json_file << "}\n";
        
        json_file.close();
        // std::cout << std::string(55, '-') << "\n";
        std::cout << "Successfully wrote output to " << filename << std::endl;
    } else {
        std::cerr << "Error: Unable to open " << filename << " for writing." << std::endl;
    }


    return 0;
}