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
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import csv
from datetime import datetime
import matplotlib.pyplot as plt

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

    # Calculate request latency statistics
    avg_actual = np.mean(actuals)
    avg_predicted = np.mean(preds)

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
    apes = (abs_errors / safe_actuals) * 100  # Absolute Percentage Error per request
    mape = np.mean(apes)

    # 4. Mean Bias: Shows systematic over-prediction (positive) or under-prediction (negative)
    mean_error = np.mean(errors)

    # 5. Tail Latency Errors: How bad are the worst 10% and 1% of predictions?
    p90_err = np.percentile(abs_errors, 90)
    p99_err = np.percentile(abs_errors, 99)

    # Print a formatted report
    print("=== Latency Prediction Accuracy Analysis ===")
    print(f"Avg Actual Latency       : {avg_actual:.3f}s")
    print(f"Avg Predicted Latency    : {avg_predicted:.3f}s")
    print("-" * 42)
    print(f"Mean Absolute Error (MAE): {mae:.3f}s")
    print(f"Mean Abs Percentage Error: {mape:.2f}%")
    print(f"Mean Error               : {mean_error:.3f}s ", end="")
    print(
        "(Simulator is OVER-predicting)"
        if mean_error > 0
        else "(Simulator is UNDER-predicting)"
    )
    print("-" * 42)
    print(f"P90 Absolute Error       : {p90_err:.3f}s")
    print(f"P99 Absolute Error       : {p99_err:.3f}s")
    print("============================================")

    # 6. Plotting
    plt.figure(figsize=(10, 6))

    # Weight each point so that the sum of all bins is 100%
    weights = np.ones_like(apes) * 100.0 / len(apes)
    
    plt.hist(apes, bins=50, weights=weights, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title('Error Frequency: APE vs Percentage of Requests', fontsize=14)
    plt.xlabel('Absolute Percentage Error (%)', fontsize=12)
    plt.ylabel('Percentage of Requests (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save the figure
    plt.savefig('latency_evaluation_plot.png', dpi=300, bbox_inches='tight')
    print("Chart saved successfully as 'latency_evaluation_plot.png'")
    
    plt.close()

    # Return as a dictionary in case you want to log these later
    return {
        "Avg_Actual": avg_actual,
        "MAE": mae,
        "MAPE": mape,
        "Mean Error": mean_error,
        "P90_Error": p90_err,
        "P99_Error": p99_err,
    }

def show_request_stats(requests, concurrencies):
    input_lens = []
    output_lens = []
    for r_id, r in requests.items():
        input_lens.append(int(r["input_tokens"]))
        output_lens.append(int(r["output_tokens"]))

    input_lens = np.array(input_lens)
    output_lens = np.array(output_lens)
    cons = np.array(concurrencies)    

    print("============ Request Statistics ============")
    print(f"Total Requests Evaluated : {len(requests)}")
    print(f"Avg Input Length         : {np.mean(input_lens):.0f} (max: {np.max(input_lens)})")
    print(f"Avg Output Length        : {np.mean(output_lens):.0f} (max: {np.max(output_lens)})")
    print(f"Avg Request Concurrency  : {np.mean(cons):.3f} (max: {np.max(cons)})")
    

def filter_requests(args):
    with open(args.request_trace_file, mode="r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # Iterate over each row
        all_requests = []
        for row in csv_reader:
            if len(row["api_base"]) > 0 and row["api_base"][-1] != "/":
                row["api_base"] += "/"

            all_requests.append(row)

    selected_requests = {}
    for r in all_requests:
        if (
            r["model"] == args.model
            and r["api_base"] == args.api_base
            and int(r["input_tokens"]) > 0
            and int(r["output_tokens"]) > 0
            and float(r["latency_ms"]) > 0
        ):
            selected_requests[r["id"]] = r
            if len(selected_requests) >= args.num_requests:
                break

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
    parser.add_argument(
        "--bench-file", 
        type=str, 
        default=None, 
        help="Results from microbenchmarks"
    )

    return parser.parse_args()

def max_overlapping_intervals(intervals):
    """
    Finds the maximum number of overlapping time intervals.
    
    :param intervals: List of lists or tuples, e.g., [[1, 4], [2, 5], [7, 9]]
    :return: Integer representing the maximum overlaps at any given time.
    """
    if not intervals:
        return 0

    events = []
    
    # 1. Separate into start and end events
    for start, end in intervals:
        events.append((start, 1))   # +1 means an interval is starting
        events.append((end, -1))    # -1 means an interval is ending
        
    # 2. Sort the events
    # We sort by time (x[0]). 
    # If times are identical, the ending event (-1) will be processed before 
    # the starting event (1). This ensures contiguous intervals like [1, 3] 
    # and [3, 5] are not counted as overlapping.
    events.sort(key=lambda x: (x[0], x[1]))
    
    max_overlaps = 0
    current_overlaps = 0
    
    # 3. Sweep through the timeline
    for time, event_type in events:
        current_overlaps += event_type
        max_overlaps = max(max_overlaps, current_overlaps)
        
    return max_overlaps


if __name__ == "__main__":
    args = parse_args()

    selected_requests = filter_requests(args)

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


    request_pool.sort(key=lambda x: x.start_time)

    # Hardware configuration
    gpu = PRESET_GPUS[args.gpu]

    # Workload configuration
    model = PRESET_MODELS[args.model]

    # Microbenchmark results
    bench_data = None
    if args.bench_file:
        with open(args.bench_file) as f:
            bench_data = json.load(f)

    # Initialize the simulation clock to the first request's arrival time
    current_time = request_pool[0].start_time if request_pool else 0.0
    waiting_queue = []
    running_queue = []
    finished_queue = []

    simulation_latency = default_latency_dict()
    concurrencies = []
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

        # 4. FORWARD PASS
        concurrencies.append(len(running_queue))
        forward_latency = estimate_forward_latency(
            running_queue, bench_data, gpu, model
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

    # Collect latency prediction results
    predicted_list = []
    actual_list = []
    for r in finished_queue:
        # Ensure every simulated request has a match in the real data
        if r.id in selected_requests:
            actual = float(selected_requests[r.id]["latency_ms"])
            predicted = r.finish_time - r.start_time
            actual_list.append(actual)
            predicted_list.append(predicted)
            error = abs(actual - predicted) / actual * 100
            print(f"actual: {actual:.3f}, predicted: {predicted:.3f}, error: {error:.1f}%, l_in: {r.input_len}, l_out: {r.output_len}")
            # print(selected_requests[r.id])
        else:
            print(f"Warning: Missing ground truth for request {r.id}")

    # Show request statistics
    show_request_stats(selected_requests, concurrencies)

    # Run the evaluation
    metrics = evaluate_latency_predictions(predicted_list, actual_list)
