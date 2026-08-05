#!/usr/bin/env python3
"""Show progress of the 30-job core reproduction queue."""

import csv
from pathlib import Path
import re


root = Path(__file__).resolve().parent
log_dir = root / "core_reproduction_logs"
manifest = log_dir / "manifest.csv"
rows = []
if manifest.exists():
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
latest_by_job = {row["job"]: row for row in rows}
complete = sum(
    row["status"] in {"complete", "skipped-complete"}
    for row in latest_by_job.values()
)
failed = sum(row["status"] == "failed" for row in latest_by_job.values())
print(f"completed={complete}/30 failed={failed}")
logs = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
if logs:
    text = logs[0].read_text(errors="replace")
    iterations = re.findall(r"iters: (\d+), epoch: (\d+)", text)
    epochs = re.findall(r"Epoch: (\d+), Steps:", text)
    state = "initializing"
    if epochs:
        state = f"epoch={epochs[-1]} complete"
    if iterations:
        state = f"epoch={iterations[-1][1]} iteration={iterations[-1][0]}"
    print(f"active_or_latest={logs[0].name} {state}")
