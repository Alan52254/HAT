#!/usr/bin/env python3
"""Extract a compact result table from the preserved DistilTS raw logs."""

from __future__ import annotations

import csv
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "core_reproduction_logs"
OUTPUT = ROOT / "results_snapshot.csv"


def last(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return ""
    value = matches[-1]
    return value if isinstance(value, str) else value[0]


def main() -> int:
    fields = [
        "job", "status", "last_epoch", "last_iteration", "mse", "mae", "rmse",
        "parameters", "inference_seconds", "gpu_memory_mb", "peak_memory_mb",
        "log_file",
    ]
    rows: list[dict[str, str]] = []
    for path in sorted(LOG_DIR.glob("*.log")):
        if path.name in {"queue_runner.log", "resource_monitor.log"}:
            continue
        text = path.read_text(errors="replace")
        metric = re.findall(
            r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+), rmse:([0-9.eE+-]+)", text
        )
        iteration = re.findall(r"iters:\s*(\d+), epoch:\s*(\d+)", text)
        completed_epochs = re.findall(r"Epoch:\s*(\d+), Steps:", text)
        mse, mae, rmse = metric[-1] if metric else ("", "", "")
        last_iteration, iteration_epoch = iteration[-1] if iteration else ("", "")
        last_epoch = completed_epochs[-1] if completed_epochs else iteration_epoch
        rows.append({
            "job": path.stem,
            "status": "complete" if metric and "Model Profiling Summary" in text else "incomplete",
            "last_epoch": last_epoch,
            "last_iteration": last_iteration,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "parameters": last(r"Total Params\s*:\s*([0-9,]+)", text),
            "inference_seconds": last(r"Inference Time \(s\)\s*:\s*([0-9.]+)", text),
            "gpu_memory_mb": last(r"GPU Mem Footprint \(MB\)\s*:\s*([0-9.]+)", text),
            "peak_memory_mb": last(r"Peak Mem \(MB\)\s*:\s*([0-9.]+)", text),
            "log_file": str(path.relative_to(ROOT)),
        })
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"wrote {OUTPUT}: {complete} complete, {len(rows) - complete} incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
