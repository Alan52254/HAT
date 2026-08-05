# DistilTS ICASSP 2026 core reproduction snapshot

This directory preserves the V100 single-seed core reproduction stopped on
2026-08-05 at the user's request.

## Snapshot status

- Formal jobs completed: 14 / 30
- Failed formal jobs: 0
- Interrupted job: `Table2_OnlyFTA_ETTm2_p192`
- Interrupted position: epoch 2, iteration 700 / 1043
- Seed: 2025
- Hardware: one NVIDIA Tesla V100-SXM2 32 GB

The interrupted job is not counted as a formal result. See
`STOPPED_STATE_2026-08-05.md` for the exact stop state.

## Contents

- `core_reproduction_logs/`: one raw log per attempted job, the queue manifest,
  runner logs, and 10-second resource telemetry in `resource_usage.csv`.
- `results_snapshot.csv`: machine-readable metrics extracted from every raw job
  log present in this snapshot.
- `REPRODUCTION_REPORT.md`: interim report and interpretation boundaries.
- `FULL_OBSERVATION_REPORT_ZH.md`: comprehensive Chinese paper/methodology,
  architecture, progress, results, resource-cost, and released-code audit report.
- `SETUP_AND_CODE_CHANGES.md`: environment setup, compatibility corrections,
  limitations, and execution notes.
- `reproduce_core.py`: resumable 30-job core queue.
- `monitor_resources.py`: GPU/CPU/RAM/disk telemetry recorder.
- `core_reproduction_status.py`: queue progress monitor.
- `requirements-reproduction.txt` and `pip_freeze.txt`: concise and exact Python
  dependency records.
- `exp/`, `models/`, `data_provider/`, `layers/`, `utils/`, `scripts/`, and
  `run.py`: source snapshot needed to reproduce the modified experiment paths.
- `local_code_changes.patch`: diff against the original DistilTS checkout.

## Reproduce or inspect

Prepare the datasets and teacher checkpoints described in
`SETUP_AND_CODE_CHANGES.md`, then run:

```bash
python reproduce_core.py
```

Monitor progress and resources with:

```bash
python core_reproduction_status.py
tail -f core_reproduction_logs/resource_usage.csv
```

Regenerate the compact result table from the raw logs with:

```bash
python summarize_results.py
```

## Files intentionally not committed

The local 192 MB snapshot archive, datasets, teacher weights, virtual
environment, and binary checkpoints are excluded from ordinary Git. They are
large generated artifacts and include files above GitHub's normal 100 MB limit.
The original local snapshot containing checkpoints is retained separately at
`/home/u4290247/DistilTS_experiment_snapshot_2026-08-05_0820`.
