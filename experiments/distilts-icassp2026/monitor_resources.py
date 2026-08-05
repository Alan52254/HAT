#!/usr/bin/env python3
"""Record system and experiment resource usage while the core queue runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "core_reproduction_logs" / "resource_usage.csv"


def experiment_processes() -> list[int]:
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "reproduce_core.py" in cmdline or ("run.py" in cmdline and "CoreReproduction" in cmdline):
            pids.append(int(entry.name))
    return pids


def process_totals(pids: list[int]) -> tuple[float, float]:
    if not pids:
        return 0.0, 0.0
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=,rss=", "-p", ",".join(map(str, pids))],
            capture_output=True, text=True, check=False,
        )
        cpu = 0.0
        rss_kib = 0
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2:
                cpu += float(fields[0])
                rss_kib += int(fields[1])
        return cpu, rss_kib / 1024
    except (OSError, ValueError):
        return 0.0, 0.0


def memory_mb() -> tuple[float, float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0
    total = values.get("MemTotal", 0) / 1024
    used = (values.get("MemTotal", 0) - values.get("MemAvailable", 0)) / 1024
    return used, total


def gpu_values() -> list[str]:
    query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return [""] * 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-when-done", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "queue_running", "process_count", "process_cpu_percent",
        "process_rss_mb", "system_load_1m", "system_ram_used_mb", "system_ram_total_mb",
        "gpu_util_percent", "gpu_memory_used_mb", "gpu_memory_total_mb",
        "gpu_temperature_c", "gpu_power_w", "disk_used_gb", "disk_free_gb",
    ]
    new_file = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        while True:
            pids = experiment_processes()
            proc_cpu, proc_rss = process_totals(pids)
            ram_used, ram_total = memory_mb()
            gpu_util, gpu_used, gpu_total, gpu_temp, gpu_power = gpu_values()
            disk = shutil.disk_usage(ROOT)
            writer.writerow({
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "queue_running": int(bool(pids)),
                "process_count": len(pids),
                "process_cpu_percent": f"{proc_cpu:.1f}",
                "process_rss_mb": f"{proc_rss:.1f}",
                "system_load_1m": f"{os.getloadavg()[0]:.2f}",
                "system_ram_used_mb": f"{ram_used:.1f}",
                "system_ram_total_mb": f"{ram_total:.1f}",
                "gpu_util_percent": gpu_util,
                "gpu_memory_used_mb": gpu_used,
                "gpu_memory_total_mb": gpu_total,
                "gpu_temperature_c": gpu_temp,
                "gpu_power_w": gpu_power,
                "disk_used_gb": f"{disk.used / 2**30:.2f}",
                "disk_free_gb": f"{disk.free / 2**30:.2f}",
            })
            if args.stop_when_done and not pids:
                break
            time.sleep(max(args.interval, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
