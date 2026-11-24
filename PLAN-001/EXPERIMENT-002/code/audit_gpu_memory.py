"""Sample whole-GPU memory while running two consecutive batches per method."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    reports, logs = args.runs / "reports", args.runs / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    csv_path = reports / f"gpu-memory-{stamp}.csv"
    report_path = reports / f"gpu-memory-{stamp}.json"
    core_path = reports / f"six-methods-memory-{stamp}.json"
    log_path = logs / f"six-methods-memory-{stamp}.log"
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("nvidia-smi was not found")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    summary = {
        "scope": "Whole GPU sampled memory, plus two-update core diagnostics",
        "sampling_interval_ms": 200,
        "csv": str(csv_path), "core_report": str(core_path), "log": str(log_path),
        "limitations": [
            "nvidia-smi memory.used includes other applications and driver/context allocations.",
            "Baseline subtraction is approximate, not exact process memory attribution.",
            "Sampling can miss short-lived peaks; allocator counters are reported separately.",
            "Tiny diagnostic models are not paper-sized training or official CLI evaluation.",
        ],
    }
    monitor = None
    started = time.time()
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as samples, log_path.open("w", encoding="utf-8") as log:
            monitor = subprocess.Popen([
                smi, "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits", "--loop-ms=200",
            ], stdout=samples, stderr=log, creationflags=creationflags)
            time.sleep(2)
            if monitor.poll() is not None:
                raise RuntimeError("nvidia-smi sampler stopped unexpectedly")
            summary["child_start_unix"] = time.time()
            command = [sys.executable, "-u", str(args.setup / "scripts/check_six_methods.py"),
                       "--repo", str(args.repo), "--setup", str(args.setup), "--runs", str(args.runs),
                       "--train-batches", "2", "--output", str(core_path)]
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            summary["child_exit_code"] = result.returncode
            summary["child_end_unix"] = time.time()
            time.sleep(2)
    finally:
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
        summary["wall_seconds"] = time.time() - started
    rows = []
    with csv_path.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            try:
                rows.append({"timestamp_local": row[0].strip(), "used_mib": float(row[1]),
                             "free_mib": float(row[2]), "utilization_percent": float(row[3])})
            except (ValueError, IndexError):
                continue
    if not rows:
        raise RuntimeError("No usable nvidia-smi samples")
    # The child starts after two seconds, leaving at least the first few samples as baseline.
    baseline = rows[:min(5, len(rows))]
    baseline_used = sum(item["used_mib"] for item in baseline) / len(baseline)
    summary.update(sample_count=len(rows), baseline_used_mib=baseline_used,
                   sampled_peak_used_mib=max(item["used_mib"] for item in rows),
                   sampled_min_free_mib=min(item["free_mib"] for item in rows),
                   sampled_peak_utilization_percent=max(item["utilization_percent"] for item in rows))
    summary["approximate_increase_over_baseline_mib"] = summary["sampled_peak_used_mib"] - baseline_used
    core = json.loads(core_path.read_text(encoding="utf-8"))
    summary["core_passed"] = core["passed"]
    summary["methods"] = [{"method": item["method"], "passed": item["passed"],
                           "batches": item.get("training_batches"), "steps": item.get("trainer_global_step"),
                           "allocator": item.get("training_cuda_memory")} for item in core["methods"]]
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(summary, memory_report=str(report_path)), ensure_ascii=True, indent=2))
    return 0 if summary["child_exit_code"] == 0 and summary["core_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
