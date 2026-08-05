# DistilTS ICASSP 2026 reproduction report

## Status

After benchmarking the full 300-job plan on the V100, execution was narrowed to
a 30-job core-method queue on 2026-08-04. It targets the evidence needed to
validate the paper's central claims: Table 2 ablations, Table 3 distillation
objectives, and Figure 3 teacher/scale comparisons. The queue is resumable and
records one log per job plus `core_reproduction_logs/manifest.csv`.

This document is an interim report while the multi-day queue runs. Final mean,
standard deviation, paper deltas, failures, and efficiency measurements must be
added only after all jobs finish.

## Hardware and software

- GPU: NVIDIA Tesla V100-SXM2, 32 GB
- Driver: 535.161.08
- Driver-reported CUDA compatibility: 12.2
- PyTorch: 2.4.1+cu121
- Python: 3.10
- Repository commit: `0f6982a9606245747f82e5101884fdfb6e3ecafd`
- Paper: arXiv:2601.12785v1

The paper used one RTX 3090 24 GB. This reproduction uses one V100 32 GB, so
wall-clock efficiency numbers are not directly comparable to the paper.

## Reproduction scope

The active queue covers:

- Table 2: Baseline, Only HW, Only FTA, and full DistilTS using iTransformer on
  ETTh1 and ETTm2 at prediction lengths 96 and 192.
- Table 3: DistilTS-L, FD-KD, and T-KD using DLinear on ETTh2 at prediction
  lengths 96 and 192.
- Figure 3: TimeMoE-50M, TimeMoE-200M, Chronos-base, and MOIRAI-base teachers
  with DLinear on ETTh1 at prediction lengths 96 and 192.
- Seed: 2025 for the first complete core pass. More seeds can be queued after
  every method path has produced one valid result.
- DLinear: 10 epochs
- iTransformer: 4 epochs
- Early stopping patience: 3
- Hidden size: 512
- Feed-forward size: 2048

The paper reports the best distilled result across teacher TSFMs. All teacher
families are therefore retained separately before selecting the best result.

## Released-code gaps

The paper's full Table 1 cannot be recreated solely from this repository:

- No TimesFM implementation, weight download, or evaluation script is included.
- No MOMENT implementation, weight download, or evaluation script is included.
- The paper reports Chronos-large, but the README and released scripts only
  prepare/use Chronos small and base.
- The paper says all experiments are repeated five times, while the released
  `run.py` originally hard-coded seed 2025. A `--seed` argument was added so the
  five repetitions are statistically meaningful.
- Several released scripts disagree with the paper's stated settings (2 or 10
  iTransformer epochs instead of 4, and `d_ff=1024` instead of 2048). The queue
  follows the paper text.

These gaps will be reported as unavailable rather than silently filled with
numbers from a different implementation.

## Code corrections required to run

- Corrected CPU/GPU device selection and CUDA profiling guards.
- Corrected `CUDA_VISIBLE_DEVICES` spelling.
- Added meaningful seed control.
- Avoided unused TimeMoE hidden-state copies when FTA is disabled. This does not
  alter model outputs or the loss.
- Isolated the project from a conflicting system PyTorch through
  `LD_LIBRARY_PATH` removal.
- Resolved the README's incompatible Chronos/Accelerate version pins.

## Preliminary validation

All three teacher adapters produced finite forecasts with the expected shape on
local checkpoints. A one-epoch CPU smoke run on ETTh1 with Chronos-Bolt-small
and DLinear completed with:

| Metric | Value |
|---|---:|
| MSE | 0.7269368 |
| MAE | 0.5675045 |
| RMSE | 0.8526059 |

This smoke result validates the pipeline only and is not treated as a paper
result because its lookback, horizon, epoch count, and hardware differ.

The first formal core job has now passed all acceptance checks:

| Job | MSE | MAE | Params | Inference | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| Table 2 Baseline, ETTh1, horizon 96 | 0.3859285 | 0.4125907 | 3,465,312 | 0.001631 s | 53.01 MB |

This is a supervised-only iTransformer baseline (`kd_alpha=0`, `vt_loss=0`);
the teacher is not instantiated. The process returned 0, wrote its checkpoint,
printed final test metrics and profiling, and was recorded as complete in the
core manifest. Total wall time was 891.1 seconds.

The first paper-scale Chronos-to-DLinear run measured approximately 235--238
seconds per training epoch before validation/test. TimeMoE autoregressive
teacher generation is slower. This measurement motivated the core-first queue;
the original 300-job five-seed matrix was projected to take at least 10--14
days, before accounting for the slowest MOIRAI jobs.

## Paper reference values

For the two DistilTS variants, Table 1 reports total-average MSE/MAE of:

| Method | MSE | MAE |
|---|---:|---:|
| DistilTS-L (DLinear) | 0.278 | 0.332 |
| DistilTS-T (iTransformer) | 0.289 | 0.335 |

The final report will compare each reproduced dataset/horizon result and the
aggregate against these values. Paper values are taken from the public preprint
at <https://arxiv.org/abs/2601.12785>.
