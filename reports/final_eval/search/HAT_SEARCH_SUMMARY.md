# HAT Search Pipeline Summary

Generated: 2026-06-09

## Formal V100-Only Update

The later formal V100-only HAT search stage was started after the predictor-only fallback stage, but it is blocked at the CUDA/V100 environment gate. The current runtime does not expose any CUDA device to PyTorch:

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
nvidia-smi = failed to communicate with NVIDIA driver
/dev/nvidia* = missing
```

Because the formal stage must use real V100 measurements only, no fallback dataset, CPU-only replacement, or predictor-only sampling was used. See:

```text
reports/final_eval/search/HAT_FORMAL_V100_SEARCH_STATUS.md
reports/final_eval/search/formal_v100_20260609_01/logs/00_cuda_v100_gate.log
```

This summary covers the attempted HAT pipeline after the completed IWSLT14 De-En SuperTransformer 50K from-scratch training. The SuperTransformer was not retrained.

## Baseline Context

| Item | Value |
| --- | --- |
| Completed SuperTransformer | IWSLT14 De-En, 50K from-scratch |
| Checkpoints | `checkpoint_best.pt`, `checkpoint_last.pt` |
| SacreBLEU | best = 32.79, last = 32.90 |
| Existing GPU latency | encoder = 4.1123 ms, decoder = 84.4480 ms, overall = 88.5603 ms |
| Existing CPU latency | overall = 193.24 ms |
| Existing FLOPs / params | 1.48G FLOPs, 30.73M params |
| Training peak GPU memory | about 25.7GB |

## Output Directory

All new artifacts from this run were written under:

```text
reports/final_eval/search/hat_pipeline_20260609_01/
```

Important files:

| Artifact | Path |
| --- | --- |
| V100 collection attempt log | `reports/final_eval/search/hat_pipeline_20260609_01/logs/01_latency_dataset_v100_50.log` |
| Fallback latency dataset copy | `reports/final_eval/search/hat_pipeline_20260609_01/latency_dataset_fallback_iwslt14deen_gpu_titanxp_all.csv` |
| Predictor log | `reports/final_eval/search/hat_pipeline_20260609_01/logs/02_latency_predictor_fallback_titanxp.log` |
| Predictor checkpoint | `reports/final_eval/search/hat_pipeline_20260609_01/predictor/latency_predictor_fallback_titanxp.pt` |
| Full evo attempt log | `reports/final_eval/search/hat_pipeline_20260609_01/logs/03_evo_search_fallback_titanxp_constraint_90.log` |
| Quick evo attempt log | `reports/final_eval/search/hat_pipeline_20260609_01/logs/04_evo_search_fallback_titanxp_constraint_90_quick.log` |
| Micro evo attempt log | `reports/final_eval/search/hat_pipeline_20260609_01/logs/05_evo_search_fallback_titanxp_constraint_90_micro.log` |
| Predictor-only top-k summary | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_candidates_summary.csv` |
| Top-k profile summary | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_candidates_profile_summary.csv` |

## 1. V100 Latency Dataset Collection

Requested target: collect a small V100 architecture-latency dataset with 50 samples.

Config:

```text
reports/final_eval/search/hat_pipeline_20260609_01/configs/latency_dataset_v100_50.yml
```

Result:

| Item | Value |
| --- | --- |
| Status | Failed |
| Intended samples | 50 |
| Output CSV | `reports/final_eval/search/hat_pipeline_20260609_01/latency_dataset_v100_50.csv` |
| Valid samples | 0 |
| Failure reason | Current runtime cannot access CUDA/V100 |
| Error | `RuntimeError: No CUDA GPUs are available` |

Because no V100 was visible to PyTorch, no new V100 architecture-latency pairs could be collected in this session.

## 2. Latency Predictor

Since V100 collection failed, I used the bundled IWSLT14 GPU latency dataset as a clearly labeled fallback:

```text
hardware-aware-transformers/latency_dataset/iwslt14deen_gpu_titanxp_all.csv
```

A copy was saved to the run directory:

```text
reports/final_eval/search/hat_pipeline_20260609_01/latency_dataset_fallback_iwslt14deen_gpu_titanxp_all.csv
```

This is **not** a newly collected V100 dataset. It is the existing IWSLT14 GPU TitanXP latency dataset bundled with the repo.

Predictor training config:

```text
reports/final_eval/search/hat_pipeline_20260609_01/configs/latency_predictor_fallback_titanxp.yml
```

Predictor metrics:

| Metric | Value |
| --- | ---: |
| Dataset rows | 2000 samples + header |
| Train steps | 1000 |
| Batch size | 128 |
| Test RMSE | 12.6486 ms |
| Test MAE | 9.5925 ms |
| Test MAPD | 0.0801 |
| Test correlation | 0.9716 |

## 3. Evolutionary Search

The original `evo_search.py` path was attempted with the trained SuperTransformer checkpoint and the fallback predictor.

| Attempt | Config | Status | Notes |
| --- | --- | --- | --- |
| Full small search | `evo_search_fallback_titanxp_constraint_90.yml` | Stopped | CPU-only validation was too slow; stopped at Iteration 0 |
| Quick search | `evo_search_fallback_titanxp_constraint_90_quick.yml` | Stopped | Still too slow in CPU-only validation |
| Micro search | `evo_search_fallback_titanxp_constraint_90_micro.yml` | Stopped | Even 1 validation batch remained too slow |

The search did load `checkpoint_best.pt` successfully, but the validation loop did not complete in a practical time under this CPU-only runtime. Therefore, no inherited validation loss from a completed evolutionary search is available.

To still produce candidate configs for downstream profiling, I generated fallback top-k configs with predictor-only sampling and ranking. This is not a full HAT evolutionary search result.

Predictor-only ranking rule:

```text
sample 500 random configs from the 50K SuperTransformer search space;
keep configs with predicted latency <= 90 ms;
rank by highest predicted latency within the constraint as a rough capacity proxy.
```

## 4. Top-k Candidate Results

Top-k configs:

| Rank | Config |
| ---: | --- |
| 1 | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_1.yml` |
| 2 | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_2.yml` |
| 3 | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_3.yml` |
| 4 | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_4.yml` |
| 5 | `reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_5.yml` |

Profile summary:

| Rank | Predicted latency | Measured GPU latency | Measured CPU latency | Params | FLOPs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 83.99 ms | N/A | 112.15 ms | 21.99M | 1.096G |
| 2 | 83.72 ms | N/A | 119.66 ms | 25.47M | 1.256G |
| 3 | 83.61 ms | N/A | 115.92 ms | 24.95M | 1.232G |
| 4 | 83.11 ms | N/A | 102.84 ms | 21.80M | 1.062G |
| 5 | 82.97 ms | N/A | 111.78 ms | 22.85M | 1.136G |

Detailed values are in:

```text
reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk/top_candidates_profile_summary.csv
```

Measured GPU latency is N/A because CUDA/V100 is unavailable in the current runtime. A GPU latency attempt for top-1 was logged and failed with:

```text
RuntimeError: No CUDA GPUs are available
```

CPU latency was measured with `--latcpu --latiter 20 --latsilent`, so it is a quick estimate rather than a high-precision latency benchmark.

## 5. Retraining / Fine-tuning

Top-1 retraining or fine-tuning was not run. The current runtime was already unable to complete even small CPU-only evolutionary validation loops, and no V100 is visible. Candidate configs were saved for later retraining on a proper GPU node.

## 6. Completion Status

| Task | Status |
| --- | --- |
| Do not retrain SuperTransformer | Done |
| Small V100 latency dataset collection | Attempted, failed due no CUDA |
| Save architecture-latency pairs | Failed for new V100 data; fallback TitanXP CSV copied |
| Train latency predictor | Completed using fallback bundled TitanXP dataset |
| Save predictor metrics | Completed |
| Run `evo_search.py` | Attempted, did not complete under CPU-only validation |
| Produce top-1/top-3/top-5 configs | Completed via predictor-only fallback |
| Profile predicted latency | Completed |
| Profile measured latency | CPU completed; GPU failed due no CUDA |
| Profile params / FLOPs | Completed for top-1 to top-5 |
| Top-1 retraining/fine-tuning | Not run |

## 7. Next Required Run On V100

For a formal HAT result, rerun the following on a node where PyTorch can see V100:

1. Re-run `latency_dataset.py` with `latency_dataset_v100_50.yml` or a larger dataset.
2. Train a predictor from the newly collected V100 CSV.
3. Re-run `evo_search.py` with the V100 predictor and a practical validation budget.
4. Measure V100 latency for the resulting top-k configs.
5. Retrain or fine-tune at least top-1 if time permits.
