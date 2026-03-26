import argparse
from configs.hw_specs import PRESET_GPUS
from configs.model_specs import PRESET_MODELS
from predictor import (
    Request,
    estimate_forward_latency,
    num_tokens_in_batch,
    default_latency_dict,
    add_latency_dicts,
    sum_latencies,
)
import numpy as np
from tqdm import tqdm
import csv
from datetime import datetime


def schedule_requests(waiting_queue, running_queue, args):
    max_num_batched_tokens = args.max_num_batched_tokens

    # empty the batch first
    for i in range(len(running_queue)):
        running_queue[i].cur_input_len = 0

    # collect new tokens by FCFS
    # top priority: decode request in running queue
    for i in range(len(running_queue)):
        if running_queue[i].input_len <= running_queue[i].kv_len:
            running_queue[i].cur_input_len = 1

    # second priority: prefill request in running queue
    for i in range(len(running_queue)):
        if running_queue[i].input_len > running_queue[i].kv_len:
            running_queue[i].cur_input_len = min(
                running_queue[i].input_len - running_queue[i].kv_len,
                max_num_batched_tokens - num_tokens_in_batch(running_queue),
            )

    # third priority: prefill request in waiting queue
    while (
        num_tokens_in_batch(running_queue) < max_num_batched_tokens
        and len(waiting_queue) > 0
    ):
        request = waiting_queue.pop(0)
        request.cur_input_len = min(
            request.input_len,
            max_num_batched_tokens - num_tokens_in_batch(running_queue),
        )
        running_queue.append(request)


def evaluate_latency_predictions(predicted_latencies, actual_latencies):
    """
    Compares predicted latencies against actual ground-truth latencies.
    Both inputs should be lists or arrays of floats (in seconds) of the same length,
    matched by request ID.
    """
    if len(predicted_latencies) != len(actual_latencies):
        raise ValueError(
            "The number of predictions must match the number of actual latencies."
        )

    if len(predicted_latencies) == 0:
        print("No data to evaluate.")
        return None

    preds = np.array(predicted_latencies)
    actuals = np.array(actual_latencies)

    # Calculate raw errors (Predicted - Actual)
    errors = preds - actuals
    abs_errors = np.abs(errors)

    # 1. Mean Absolute Error (MAE): Average error in seconds
    mae = np.mean(abs_errors)

    # 2. Root Mean Square Error (RMSE): Penalizes larger prediction errors heavily
    rmse = np.sqrt(np.mean(errors**2))

    # 3. Mean Absolute Percentage Error (MAPE): Error relative to request length
    # (Avoid division by zero in case of anomalous 0-second actuals)
    safe_actuals = np.where(actuals == 0, 1e-9, actuals)
    mape = np.mean(abs_errors / safe_actuals) * 100

    # 4. Mean Bias: Shows systematic over-prediction (positive) or under-prediction (negative)
    mean_bias = np.mean(errors)

    # 5. Tail Latency Errors: How bad are the worst 10% and 1% of predictions?
    p90_err = np.percentile(abs_errors, 90)
    p99_err = np.percentile(abs_errors, 99)

    # Print a formatted report
    print("=== Latency Prediction Accuracy Analysis ===")
    print(f"Total Requests Evaluated : {len(preds)}")
    print("-" * 42)
    print(f"Mean Absolute Error (MAE): {mae:.4f}s")
    print(f"Root Mean Square Error   : {rmse:.4f}s")
    print(f"Mean Abs Percentage Error: {mape:.2f}%")
    print(f"Mean Bias                : {mean_bias:.4f}s ", end="")
    print(
        "(Simulator is OVER-predicting)"
        if mean_bias > 0
        else "(Simulator is UNDER-predicting)"
    )
    print("-" * 42)
    print(f"P90 Absolute Error       : {p90_err:.4f}s")
    print(f"P99 Absolute Error       : {p99_err:.4f}s")
    print("============================================")

    # Return as a dictionary in case you want to plot or log these later
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Mean Bias": mean_bias,
        "P90_Error": p90_err,
        "P99_Error": p99_err,
    }


def filter_requests(path, api_base):
    with open(path, mode="r", newline="", encoding="utf-8") as csvfile:
        # Create a dictionary reader object
        csv_reader = csv.DictReader(csvfile)

        # Iterate over each row (each row is a dictionary)
        all_requests = []
        for row in csv_reader:
            if len(row["api_base"]) > 0 and row["api_base"][-1] != "/":
                row["api_base"] += "/"

            all_requests.append(row)

    selected_requests = {}
    for r in all_requests:
        if (
            r["model"] == "openai/gpt-oss-20b"
            and r["api_base"] == api_base
            and int(r["input_tokens"]) > 0
            and int(r["output_tokens"]) > 0
        ):
            selected_requests[r["id"]] = r

    return selected_requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure hardware and workload for benchmarking/inference."
    )

    # Hardware configuration
    parser.add_argument(
        "--gpu",
        type=str,
        default="h100-sxm",
        help="GPU preset name to use (default: h100-sxm)",
    )

    # Workload configuration
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-30b",
        help="Model preset name to use (default: qwen3-30b)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Total number of requests (default: 12800)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum number of batched tokens (default: 8192)",
    )
    parser.add_argument(
        "--request-trace-file",
        type=str,
        default=None,
        help="The csv file of request traces for evaluation",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="The api base for evaluation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    selected_requests = filter_requests(args.request_trace_file, args.api_base)

    request_pool = []
    for r in selected_requests.values():
        start_time = datetime.fromisoformat(
            r["startTime"].replace("Z", "+00:00")
        ).timestamp()
        request_pool.append(
            Request(
                start_time, int(r["input_tokens"]), int(r["output_tokens"]), r["id"]
            )
        )

        if len(request_pool) >= args.num_requests:
            break

    request_pool.sort(key=lambda x: x.start_time)

    # Hardware configuration
    gpu = PRESET_GPUS[args.gpu]

    # Workload configuration
    model = PRESET_MODELS[args.model]


    # Initialize the simulation clock to the first request's arrival time
    current_time = request_pool[0].start_time if request_pool else 0.0
    waiting_queue = []
    running_queue = []
    finished_queue = []

    simulation_latency = default_latency_dict()
    pbar = tqdm(total=len(request_pool))

    # Expanded the while condition to ensure the loop doesn't end if requests are still arriving or waiting
    while (
        len(request_pool) > 0
        or len(waiting_queue) > 0
        or len(running_queue) > 0
    ):

        # 1. FAST-FORWARD TIME (System is idle)
        # If no requests are active, jump the clock directly to the next request's arrival time
        if (
            len(running_queue) == 0
            and len(waiting_queue) == 0
            and len(request_pool) > 0
        ):
            if current_time < request_pool[0].start_time:
                current_time = request_pool[0].start_time

        # 2. ADMIT ARRIVING REQUESTS
        # Move requests into the system if their start time has passed
        while len(request_pool) > 0 and request_pool[0].start_time <= current_time:
            waiting_queue.append(request_pool.pop(0))

        # 3. SCHEDULE
        # Give the scheduler only the requests that have actually arrived
        schedule_requests(waiting_queue, running_queue, args)

        # Edge Case: If the scheduler couldn't run anything (e.g. waiting for memory to free up)
        # tick the clock forward slightly to prevent an infinite loop.
        if len(running_queue) == 0:
            current_time += 0.001  # 1 millisecond tick (adjust based on your scale)
            continue

        # 4. FORWARD PASS
        forward_latency = estimate_forward_latency(
            running_queue, gpu, model
        )
        add_latency_dicts(simulation_latency, forward_latency)

        # Get the actual time (in seconds/ms) this step took
        step_duration = sum_latencies(forward_latency)

        # Advance the global clock
        current_time += step_duration.bound

        # 5. UPDATE REQUEST STATUS
        unfinished_id = []
        for i in range(len(running_queue)):
            req = running_queue[i]

            # update kv cache length
            req.kv_len += req.cur_input_len

            # update latency
            if req.input_len >= req.kv_len:
                add_latency_dicts(req.latency.prefill, forward_latency)
            else:
                add_latency_dicts(req.latency.decode, forward_latency)

            # delete finished request
            if req.kv_len != req.input_len + req.output_len - 1:
                unfinished_id.append(i)
            else:
                # RECORD END-TO-END LATENCY
                req.finish_time = current_time
                # req.latency.e2e = req.finish_time - req.start_time
                finished_queue.append(req)
                pbar.update(1)

        running_queue = [running_queue[i] for i in unfinished_id]

        # 6. UPDATE WAITING LATENCIES
        for i in range(len(waiting_queue)):
            waiting_queue[i].latency.waiting += step_duration

    pbar.close()

    predicted_list = []
    actual_list = []
    for r in finished_queue:
        # Ensure every simulated request has a match in the real data
        if r.id in selected_requests:
            predicted_list.append(r.finish_time - r.start_time)
            actual_list.append(float(selected_requests[r.id]["latency_ms"]))

        else:
            print(f"Warning: Missing ground truth for request {r.id}")

    # Run the evaluation
    metrics = evaluate_latency_predictions(predicted_list, actual_list)
