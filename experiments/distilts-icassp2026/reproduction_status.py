#!/usr/bin/env python3
"""Print concise progress for the repository-supported reproduction queue."""

from pathlib import Path
import csv
import re


root = Path(__file__).resolve().parent
log_dir = root / "reproduction_logs"
manifest = log_dir / "manifest.csv"
rows = []
if manifest.exists():
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

latest = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
latest = [path for path in latest if path.name != "queue.log"]
complete = sum(row["status"] in {"complete", "skipped-complete"} for row in rows)
failed = sum(row["status"] == "failed" for row in rows)
print(f"completed={complete}/300 failed={failed}")
if latest:
    text = latest[0].read_text(errors="replace")
    epochs = re.findall(r"Epoch: (\d+), Steps:.*", text)
    iterations = re.findall(r"iters: (\d+), epoch: (\d+)", text)
    progress = f"epoch={epochs[-1]} complete" if epochs else "initializing"
    if iterations:
        progress = f"epoch={iterations[-1][1]} iteration={iterations[-1][0]}"
    print(f"active_or_latest={latest[0].name} {progress}")
