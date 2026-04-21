# set library path for CUPTI
CUDA_HOME=/usr/local/cuda
CUPTI_INCLUDE=$CUDA_HOME/extras/CUPTI/include
CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64

# compute microbenchmarks
python3 compute_perf.py

# memory microbenchmarks
nvcc -I$CUPTI_INCLUDE -L$CUPTI_LIB -lcupti memory_perf.cu -o memory_perf
./memory_perf

# find p, aggregate results
python3 find_p.py

# clean file
rm memory_perf
rm compute.json
rm memory.json
