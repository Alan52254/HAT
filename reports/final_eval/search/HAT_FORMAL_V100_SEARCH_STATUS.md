# Formal V100 HAT Search Status

Generated: 2026-06-09

This file records the current formal HAT Evolutionary Search attempt after the predictor-only fallback stage. The fallback dataset and predictor-only candidates are treated as previous exploratory artifacts only; they are **not** used for this formal V100-only stage.

## Policy For This Stage

| Requirement | Status |
| --- | --- |
| Do not retrain the completed SuperTransformer | Enforced |
| Do not use fallback latency dataset | Enforced |
| Do not use predictor-only fallback sampling | Enforced |
| Require real CUDA/V100 visibility before continuing | Enforced |

## CUDA / V100 Gate

The formal pipeline is currently blocked at the environment gate.

| Check | Result |
| --- | --- |
| PyTorch version | 1.7.1 |
| PyTorch CUDA build | 10.2 |
| `torch.cuda.is_available()` | `False` |
| `torch.cuda.device_count()` | `0` |
| `nvidia-smi` | Fails to communicate with NVIDIA driver |
| `/dev/nvidia*` | Not present |

Detailed log:

```text
reports/final_eval/search/formal_v100_20260609_01/logs/00_cuda_v100_gate.log
```

## Execution Status

| Step | Status | Notes |
| --- | --- | --- |
| 1. Ensure PyTorch can call CUDA/V100 | Blocked | Current runtime exposes no CUDA devices |
| 2. Collect V100 architecture-latency dataset | Not run | Requires successful CUDA/V100 gate |
| 3. Train V100 latency predictor | Not run | Requires real V100 dataset |
| 4. Run formal evolutionary search | Not run | Requires V100 predictor and feasible inherited validation |
| 5. Report top-k candidates | Not available | Requires completed formal search |

## Blocker

This is an infrastructure/runtime blocker, not a SuperTransformer training issue and not an HAT configuration issue. The current container or node does not expose NVIDIA GPU devices:

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
nvidia-smi = failed to communicate with NVIDIA driver
/dev/nvidia* = missing
```

Because the user explicitly requested real V100 measurements only, the formal pipeline must stop here rather than falling back to TitanXP data, CPU measurements, or predictor-only sampling.

## Ready To Continue On A V100 Node

Once running on a node where `nvidia-smi` shows a V100 and PyTorch reports `torch.cuda.is_available() == True`, continue with:

1. Run `latency_dataset.py` using a V100-specific config and save the output under `reports/final_eval/search/formal_v100_20260609_01/`.
2. Train `latency_predictor.py` using only the newly collected V100 dataset.
3. Run `evo_search.py` with the trained V100 predictor and `checkpoint_best.pt`, ensuring inherited validation loss is evaluated.
4. Profile the resulting top-k candidates on V100 for measured latency, FLOPs, and parameters.
5. Update `HAT_SEARCH_SUMMARY.md` with predicted V100 latency, measured V100 latency, inherited validation loss or PPL, FLOPs, and parameters.
