import os

# Keep this Stage 1 benchmark sequential at the Python/numerical-library level.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import platform
import socket
import statistics
import time
from pathlib import Path

import pandas as pd
import psutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARTI 503 Stage 1 - Sequential benchmark for NYC Taxi train.csv"
    )
    parser.add_argument(
        "--file",
        default="train.csv",
        help="Path to train.csv (default: train.csv in the same folder).",
    )
    parser.add_argument(
        "--rows",
        default="10000,100000,500000,1000000",
        help="Comma-separated input sizes, e.g. 10000,100000,500000,1000000",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated measurements for each input size (default: 3).",
    )
    return parser.parse_args()


def cpu_info():
    return {
        "computer_name": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Not reported by OS",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
    }


def run_sequential_analysis(df):
    """Run the Stage 1 analytical workload sequentially and return timings/results."""
    timings = {}
    results = {}

    start = time.perf_counter()
    results["trip_duration_average"] = float(df["trip_duration"].mean())
    results["trip_duration_min"] = int(df["trip_duration"].min())
    results["trip_duration_max"] = int(df["trip_duration"].max())
    timings["avg_min_max"] = time.perf_counter() - start

    start = time.perf_counter()
    # Frequency counts required by the project plan.
    vendor_counts = df["vendor_id"].value_counts(sort=False)
    passenger_counts = df["passenger_count"].value_counts(sort=False)
    results["vendor_count_groups"] = int(len(vendor_counts))
    results["passenger_count_groups"] = int(len(passenger_counts))
    timings["frequency_counts"] = time.perf_counter() - start

    start = time.perf_counter()
    grouped = df.groupby("vendor_id", sort=False)["trip_duration"].mean()
    results["vendor_group_average_count"] = int(len(grouped))
    timings["grouped_aggregation"] = time.perf_counter() - start

    timings["analysis_total"] = sum(timings.values())
    return timings, results


def main():
    args = parse_args()
    csv_path = Path(args.file)

    if not csv_path.exists():
        raise SystemExit(
            f"Could not find {csv_path}. Put train.csv in the same folder as this script."
        )

    requested_sizes = []
    for value in args.rows.split(","):
        value = value.strip()
        if value:
            requested_sizes.append(int(value))
    if not requested_sizes:
        raise SystemExit("No valid input sizes were provided.")
    if args.repeats < 2:
        raise SystemExit("Use at least 2 repeats so the benchmark is not based on one run.")

    max_rows = max(requested_sizes)
    print("Loading train.csv once (loading time is not included in analysis timing)...")
    load_start = time.perf_counter()
    data = pd.read_csv(csv_path, nrows=max_rows)
    load_time = time.perf_counter() - load_start

    available_rows = len(data)
    sizes = [s for s in requested_sizes if s <= available_rows]
    if not sizes:
        raise SystemExit(f"Requested sizes exceed the available {available_rows:,} rows.")

    info = cpu_info()
    print("\nSystem information")
    print("------------------")
    print(f"Computer: {info['computer_name']}")
    print(f"CPU: {info['cpu']}")
    print(f"Physical cores: {info['physical_cores']}")
    print(f"Logical processors: {info['logical_processors']}")
    print(f"Rows loaded: {available_rows:,}")
    print(f"CSV load time (not benchmarked): {load_time:.4f} s")

    raw_rows = []
    summary_rows = []

    for size in sizes:
        df = data.iloc[:size]
        print(f"\nBenchmarking {size:,} rows ({args.repeats} repeated runs)...")
        repeat_totals = []
        operation_samples = {
            "avg_min_max": [],
            "frequency_counts": [],
            "grouped_aggregation": [],
        }
        last_results = None

        for run_number in range(1, args.repeats + 1):
            timings, results = run_sequential_analysis(df)
            last_results = results
            repeat_totals.append(timings["analysis_total"])
            for op in operation_samples:
                operation_samples[op].append(timings[op])

            raw_rows.append({
                "computer_name": info["computer_name"],
                "cpu": info["cpu"],
                "physical_cores": info["physical_cores"],
                "logical_processors": info["logical_processors"],
                "input_rows": size,
                "run": run_number,
                "avg_min_max_seconds": timings["avg_min_max"],
                "frequency_counts_seconds": timings["frequency_counts"],
                "grouped_aggregation_seconds": timings["grouped_aggregation"],
                "analysis_total_seconds": timings["analysis_total"],
            })
            print(f"  Run {run_number}: {timings['analysis_total']:.6f} s")

        avg_total = statistics.mean(repeat_totals)
        op_means = {op: statistics.mean(vals) for op, vals in operation_samples.items()}
        slowest_operation = max(op_means, key=op_means.get)

        summary_rows.append({
            "computer_name": info["computer_name"],
            "cpu": info["cpu"],
            "physical_cores": info["physical_cores"],
            "logical_processors": info["logical_processors"],
            "input_rows": size,
            "repeats": args.repeats,
            "average_execution_time_seconds": avg_total,
            "slowest_operation": slowest_operation,
            "slowest_operation_average_seconds": op_means[slowest_operation],
        })

        print(f"  Average: {avg_total:.6f} s")
        print(f"  Slowest operation: {slowest_operation} ({op_means[slowest_operation]:.6f} s avg)")
        if last_results:
            print(f"  Trip duration avg/min/max: "
                  f"{last_results['trip_duration_average']:.2f} / "
                  f"{last_results['trip_duration_min']} / "
                  f"{last_results['trip_duration_max']}")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in info["computer_name"])
    raw_file = Path(f"benchmark_runs_{safe_name}.csv")
    summary_file = Path(f"benchmark_summary_{safe_name}.csv")
    system_file = Path(f"system_info_{safe_name}.txt")

    with raw_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with system_file.open("w", encoding="utf-8") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        f.write(f"csv_file: {csv_path.name}\n")
        f.write(f"rows_loaded: {available_rows}\n")
        f.write(f"repeats: {args.repeats}\n")
        f.write(f"input_sizes: {','.join(map(str, sizes))}\n")

    print("\nDone. Send these generated files to the group leader:")
    print(f"- {summary_file}")
    print(f"- {raw_file}")
    print(f"- {system_file}")


if __name__ == "__main__":
    main()
