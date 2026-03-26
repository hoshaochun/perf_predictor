# Performance Predictor for LLM Inference
## Installation
```
uv pip install -r requirements.txt
```
## Current Support
* hardware: "4090", "h100-sxm", "h200-nvl"
* model: "gpt-oss-20b", "qwen3-30b" (for qwen3-coder-30b-a3b)
## Run Predictor
Example Usage:
```python=
python3 run_predictor.py \
    --gpu "h100-sxm" \
    --model "qwen3-30b" \
    --num-requests 6400 \
    --num-concurrency 256 \
    --input-len 128 \
    --output-len 128 \
    --random-range-ratio 0.1 \
    --max-num-batched-tokens 8192
```
## Run Evaluation
Example Usage
```python=
python3 run_evaluation.py \
    --gpu "h200-nvl" \
    --model "gpt-oss-20b" \
    --num-requests 1000 \
    --max-num-batched-tokens 8192 \
    --request-trace-file [path to trace file] \
    --api-base [api base for evaluation]
```
