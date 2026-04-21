# Performance Predictor for LLM Inference
## Installation
```
uv pip install -r requirements.txt
```
## Current Support
* hardware: "4090", "h100-sxm", "h200-nvl", "b200"
* model: "openai/gpt-oss-20b", "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
## Run Predictor
Example Usage:
```
python3 run_predictor.py \
    --gpu "h100-sxm" \
    --model "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit" \
    --num-requests 6400 \
    --num-concurrency 256 \
    --input-len 128 \
    --output-len 128 \
    --random-range-ratio 0.1 \
    --max-num-batched-tokens 8192 \
    --bench-file [microbenchmark result file]
```
## Run Evaluation
Example Usage:
```
python3 run_evaluation.py \
    --gpu "h200-nvl" \
    --model "openai/gpt-oss-20b" \
    --num-requests 1000 \
    --max-num-batched-tokens 8192 \
    --request-trace-file [path to trace file] \
    --api-base [api base for evaluation] \
    --bench-file [microbenchmark result file]
```
