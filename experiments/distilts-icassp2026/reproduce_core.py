#!/usr/bin/env python3
"""Resumable V100 queue for the paper's core claims and ablations."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "core_reproduction_logs"
MANIFEST = LOG_DIR / "manifest.csv"
SEED = 2025


def is_complete(job: str, path: Path) -> bool:
    if not path.exists():
        return False
    tail = path.read_text(errors="replace")[-20000:]
    has_final_output = "mse:" in tail and "Model Profiling Summary" in tail
    has_checkpoint = any((ROOT / "checkpoints").glob(f"*core_{job}*/checkpoint.pth"))
    return has_final_output and has_checkpoint


def record(job: str, status: str, code: int, seconds: float, log: Path) -> None:
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["job", "status", "returncode", "seconds", "log"])
        writer.writerow([job, status, code, f"{seconds:.1f}", log])


def common(job: str, model: str, dataset: tuple, seq_len: int, pred_len: int) -> list[str]:
    ds_name, root_path, data_path, channels, data_flag = dataset
    return [
        str(ROOT / ".venv/bin/python"), "-u", "run.py",
        "--seed", str(SEED), "--is_training", "1",
        "--root_path", root_path, "--data_path", data_path,
        "--model_id", f"core_{job}", "--model", model,
        "--data", data_flag, "--features", "M",
        "--seq_len", str(seq_len), "--label_len", "0",
        "--pred_len", str(pred_len), "--d_model", "512",
        "--d_ff", "2048", "--e_layers", "1", "--d_layers", "1",
        "--factor", "3", "--enc_in", str(channels),
        "--dec_in", str(channels), "--c_out", str(channels),
        "--des", "CoreReproduction", "--patience", "3",
        "--gen_chunk", "128", "--itr", "1",
    ]


def build_jobs() -> list[tuple[str, list[str]]]:
    etth1 = ("ETTh1", "./dataset/ETT-small/", "ETTh1.csv", 7, "ETTh1")
    etth2 = ("ETTh2", "./dataset/ETT-small/", "ETTh2.csv", 7, "ETTh2")
    ettm2 = ("ETTm2", "./dataset/ETT-small/", "ETTm2.csv", 7, "ETTm2")
    jobs: list[tuple[str, list[str]]] = []

    # Table 2: iTransformer ablation with TimeMoE-50M.
    variants = [
        ("Baseline", "0", "0", "0"),
        ("OnlyHW", "0.5", "0", "1"),
        ("OnlyFTA", "0.5", "1", "0"),
        ("FullDistilTS", "0.5", "1", "1"),
    ]
    for dataset in (etth1, ettm2):
        for pred_len, seq_len in ((96, 512), (192, 1024)):
            for name, kd_alpha, vt_loss, tau in variants:
                job = f"Table2_{name}_{dataset[0]}_p{pred_len}"
                cmd = common(job, "iTransformer", dataset, seq_len, pred_len)
                cmd += [
                    "--task_name", "Exp_DistilTS",
                    "--pretrained_path", "./TimeMoE-50M",
                    "--TSFModel", "TimeMoe", "--kd_alpha", kd_alpha,
                    "--vt_loss", vt_loss, "--horizon_weight_tau", tau,
                    "--train_epochs", "4",
                ]
                jobs.append((job, cmd))

    # Table 3: DLinear comparison of distillation objectives on ETTh2.
    methods = [
        ("DistilTSL", "Exp_DistilTS"),
        ("FDKD", "exp_DistilTS_FDKD"),
        ("TKD", "exp_DistilTS_TKD"),
    ]
    for pred_len, seq_len in ((96, 512), (192, 1024)):
        for name, task in methods:
            job = f"Table3_{name}_ETTh2_p{pred_len}"
            cmd = common(job, "DLinear", etth2, seq_len, pred_len)
            cmd += [
                "--task_name", task, "--pretrained_path", "./TimeMoE-50M",
                "--TSFModel", "TimeMoe", "--kd_alpha", "0.5",
                "--vt_loss", "0", "--horizon_weight_tau", "1",
                "--train_epochs", "10",
            ]
            jobs.append((job, cmd))

    # Figure 3: teacher architecture and TimeMoE scale, DLinear on ETTh1.
    teachers = [
        ("TimeMoE50M", "TimeMoe", "./TimeMoE-50M", (512, 1024)),
        ("TimeMoE200M", "TimeMoe", "./TimeMoE-200M", (512, 1024)),
        ("ChronosBase", "Chronos", "./chronos-bolt-base", (512, 512)),
        ("MoiraiBase", "Moirai", "./MOIRAI-base", (5000, 5000)),
    ]
    for label, teacher, path, seq_lens in teachers:
        for pred_len, seq_len in zip((96, 192), seq_lens):
            job = f"Figure3_{label}_DLinear_ETTh1_p{pred_len}"
            cmd = common(job, "DLinear", etth1, seq_len, pred_len)
            cmd += [
                "--task_name", "Exp_DistilTS", "--pretrained_path", path,
                "--TSFModel", teacher, "--kd_alpha", "0.5",
                "--vt_loss", "0", "--horizon_weight_tau", "1",
                "--train_epochs", "10",
            ]
            jobs.append((job, cmd))
    return jobs


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    for job, command in build_jobs():
        log = LOG_DIR / f"{job}.log"
        if is_complete(job, log):
            record(job, "skipped-complete", 0, 0, log)
            continue
        start = time.monotonic()
        with log.open("w") as handle:
            proc = subprocess.run(
                command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        seconds = time.monotonic() - start
        status = "complete" if proc.returncode == 0 and is_complete(job, log) else "failed"
        record(job, status, proc.returncode, seconds, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
