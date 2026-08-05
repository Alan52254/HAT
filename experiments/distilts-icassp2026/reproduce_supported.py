#!/usr/bin/env python3
"""Run the paper's repository-supported DistilTS main-result matrix.

The queue is resumable: a job is skipped only after its log contains the final
profiling summary. Failed jobs are recorded and the queue continues.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "reproduction_logs"
MANIFEST = LOG_DIR / "manifest.csv"

DATASETS = [
    ("ETTh1", "./dataset/ETT-small/", "ETTh1.csv", 7, "ETTh1"),
    ("ETTh2", "./dataset/ETT-small/", "ETTh2.csv", 7, "ETTh2"),
    ("ETTm1", "./dataset/ETT-small/", "ETTm1.csv", 7, "ETTm1"),
    ("ETTm2", "./dataset/ETT-small/", "ETTm2.csv", 7, "ETTm2"),
    ("Weather", "./dataset/weather/", "weather.csv", 21, "custom"),
]

TEACHERS = {
    "TimeMoe": ("./TimeMoE-50M", (512, 1024)),
    "Chronos": ("./chronos-bolt-base", (512, 512)),
    "Moirai": ("./MOIRAI-base", (5000, 5000)),
}


def completed(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    tail = log_path.read_text(errors="replace")[-8000:]
    return "Model Profiling Summary" in tail


def append_manifest(row: list[object]) -> None:
    new_file = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["job", "status", "returncode", "seconds", "log"])
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-at", default="")
    args = parser.parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    started = not args.start_at

    for repeat in range(1, args.repeats + 1):
        seed = 2024 + repeat
        for teacher, (teacher_path, seq_lens) in TEACHERS.items():
            for student in ("DLinear", "iTransformer"):
                epochs = 10 if student == "DLinear" else 4
                vt_loss = int(teacher == "TimeMoe" and student == "iTransformer")
                for ds_name, root_path, data_path, channels, data_flag in DATASETS:
                    for pred_len, seq_len in zip((96, 192), seq_lens):
                        job = f"{teacher}_{student}_{ds_name}_s{seq_len}_p{pred_len}_r{repeat}"
                        if not started:
                            started = job == args.start_at
                            if not started:
                                continue
                        log_path = LOG_DIR / f"{job}.log"
                        if completed(log_path):
                            append_manifest([job, "skipped-complete", 0, 0, log_path])
                            continue

                        command = [
                            str(ROOT / ".venv/bin/python"), "-u", "run.py",
                            "--seed", str(seed),
                            "--task_name", "Exp_DistilTS",
                            "--is_training", "1",
                            "--pretrained_path", teacher_path,
                            "--root_path", root_path,
                            "--data_path", data_path,
                            "--model_id", f"repro_{job}",
                            "--model", student,
                            "--data", data_flag,
                            "--features", "M",
                            "--seq_len", str(seq_len),
                            "--label_len", "0",
                            "--pred_len", str(pred_len),
                            "--d_model", "512",
                            "--d_ff", "2048",
                            "--e_layers", "1",
                            "--d_layers", "1",
                            "--factor", "3",
                            "--enc_in", str(channels),
                            "--dec_in", str(channels),
                            "--c_out", str(channels),
                            "--des", "PaperReproduction",
                            "--vt_loss", str(vt_loss),
                            "--gen_chunk", "128",
                            "--train_epochs", str(epochs),
                            "--patience", "3",
                            "--TSFModel", teacher,
                            "--itr", "1",
                        ]
                        env = os.environ.copy()
                        env.pop("LD_LIBRARY_PATH", None)
                        started_at = time.monotonic()
                        with log_path.open("w") as log:
                            proc = subprocess.run(
                                command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False,
                            )
                        elapsed = time.monotonic() - started_at
                        status = "complete" if proc.returncode == 0 and completed(log_path) else "failed"
                        append_manifest([job, status, proc.returncode, f"{elapsed:.1f}", log_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
