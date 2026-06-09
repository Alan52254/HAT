# HAT IWSLT14 De-En Experiment Results

Generated: 2026-06-09

This file consolidates the experiment results currently available in this workspace. It uses the raw logs and summary files as sources of truth. Some older summary files are stale, so this report separates completed measurements from failed or skipped attempts.

## 1. Experiment Scope

| Item | Value |
| --- | --- |
| Project | Hardware-Aware Transformers (HAT) |
| Task | IWSLT14 De-En translation |
| Dataset path | `hardware-aware-transformers/data/binary/iwslt14_de_en` |
| Main trained model | IWSLT14 SuperTransformer, from-scratch, 50K updates |
| Main checkpoint dir | `checkpoints/iwslt14_super_full50000_fromscratch` |
| Main evaluation SubTransformer config | `configs/iwslt14.de-en/subtransformer/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.yml` |
| Repo commit recorded in final eval | `70e5a279d080670208249fdd98ed731fa9bcc466` |

## 2. Environment

Final evaluation environment recorded in `reports/final_eval/00_environment_check.log`:

| Item | Value |
| --- | --- |
| Timestamp | Tue Jun 9 00:45:24 CST 2026 |
| Host | `y0ny9xctr1780243117444-2b2bl` |
| Working dir | `/home/u4290247/HAT/hardware-aware-transformers` |
| Python | `/home/u4290247/miniconda3/envs/hat/bin/python` |
| Python version | 3.7.12 |
| PyTorch | 1.7.1 |
| PyTorch CUDA | 10.2 |
| CUDA available to PyTorch | False |
| CUDA device count | 0 |
| `nvidia-smi` in final eval env | Failed, NVIDIA driver unavailable |

Note: GPU training and GPU latency logs exist from earlier V100 runs, but the final evaluation environment used on 2026-06-09 could not access CUDA.

## 3. Runtime Compatibility Patches

The following runtime compatibility fixes were applied or verified. They do not change model architecture or checkpoint weights.

| File | Status |
| --- | --- |
| `fairseq/modules/multihead_attention_super.py` | Existing in-place scaling fix verified: `q *= self.scaling` changed to `q = q * self.scaling` |
| `fairseq/search.py` | Fixed PyTorch division compatibility: `torch.div(..., out=LongTensor)` changed to `torch.floor_divide(...)` |
| `score.py` | Fixed SacreBLEU input compatibility: pass list of stripped strings instead of file objects |

Patch artifacts are in `reports/final_eval/patches/`.

## 4. Training Runs

### 4.1 Failed Initial Smoke Run

Source: `reports/supertrain/02_super_smoke100.log`

| Item | Value |
| --- | --- |
| Run | Initial SuperTransformer smoke test |
| Intended updates | 100 |
| Status | Failed |
| Failure | In-place gradient modification error |
| Error summary | `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` |

This led to the `multihead_attention_super.py` in-place scaling compatibility fix.

### 4.2 Smoke 100 After Patch

Source: `reports/supertrain/03_super_smoke100_after_inplace_patch.log`

| Item | Value |
| --- | ---: |
| Status | Completed |
| Updates | 100 |
| Training time | 32.4 sec |
| Final train loss | 12.990 |
| Final train nll_loss | 12.951 |
| Final train ppl | 7918.46 |
| OOM | 0 |
| SuperTransformer non-embedding params | 55,160,064 |
| Embedding params | 14,151,680 |
| Estimated total params | 69,311,744 |

Validation at 100 updates:

| SubTransformer | Loss | NLL loss | PPL | Updates |
| --- | ---: | ---: | ---: | ---: |
| largest_arbitrary1 | 12.405 | 12.291 | 5009.91 | 100 |
| smallest_arbitrary1 | 12.906 | 12.866 | 7464.56 | 100 |

### 4.3 Smoke 1000

Source: `reports/supertrain/04_super_smoke1000.log`

| Item | Value |
| --- | ---: |
| Status | Completed |
| Updates | 1000 |
| Training time | 103.6 sec |
| Final train loss | 10.591 |
| Final train nll_loss | 10.221 |
| Final train ppl | 1193.65 |
| OOM | 0 |
| SuperTransformer non-embedding params | 55,160,064 |
| Embedding params | 14,151,680 |
| Estimated total params | 69,311,744 |

Validation at 1000 updates:

| SubTransformer | Loss | NLL loss | PPL | Updates |
| --- | ---: | ---: | ---: | ---: |
| largest_arbitrary1 | 9.626 | 8.981 | 505.36 | 1000 |
| smallest_arbitrary1 | 9.361 | 8.793 | 443.61 | 1000 |

### 4.4 Full 50K From-Scratch SuperTransformer

Sources:

- `reports/supertrain/05_super_full50000_fromscratch.log`
- `final_artifacts/50k_training_summary.txt`
- `reports/final_eval/01_checkpoint_check.log`

| Item | Value |
| --- | ---: |
| Status | Completed |
| Updates | 50,000 |
| Final epoch | 46 |
| Training time | 28,627.2 sec |
| Final train loss | 3.772 |
| Final train nll_loss | 2.337 |
| Final train ppl | 5.05 |
| Final lr | 0.000223607 |
| Final gradient norm | 1.138 |
| OOM | 0 |
| SuperTransformer non-embedding params | 55,160,064 |
| Embedding params | 14,151,680 |
| Estimated total params | 69,311,744 |

Final validation at 50,000 updates:

| SubTransformer | Loss | NLL loss | PPL | Updates | Best loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| largest_arbitrary1 | 4.019 | 2.448 | 5.46 | 50,000 | 4.01913 |
| smallest_arbitrary1 | 4.434 | 2.891 | 7.42 | 50,000 | 4.42398 |

Near-final validation trajectory:

| Update | largest loss | largest PPL | smallest loss | smallest PPL |
| ---: | ---: | ---: | ---: | ---: |
| 45,141 | 4.037 | 5.52 | 4.470 | 7.62 |
| 46,242 | 4.025 | 5.46 | 4.435 | 7.44 |
| 47,343 | 4.031 | 5.50 | 4.450 | 7.49 |
| 48,444 | 4.028 | 5.51 | 4.448 | 7.49 |
| 49,545 | 4.013 | 5.44 | 4.424 | 7.38 |
| 50,000 | 4.019 | 5.46 | 4.434 | 7.42 |

Checkpoint inventory:

| Item | Value |
| --- | --- |
| Checkpoint directory size | 20G |
| Important checkpoints | `checkpoint_best.pt`, `checkpoint_last.pt`, `checkpoint_46_50000.pt` |
| Checkpoint file size | about 794M each |
| Archived final artifacts | `final_artifacts/iwslt14_super_50k_checkpoint_best.pt`, `final_artifacts/iwslt14_super_50k_checkpoint_last.pt`, `final_artifacts/iwslt14_super_50k_checkpoint_46_50000.pt` |

## 5. Translation Quality: BLEU / SacreBLEU

The old BLEU attempt in `reports/profile_50k/04_bleu_checkpoint_best.log` failed before compatibility fixes because of:

- `fairseq/search.py`: `RuntimeError: result type Float can't be cast to the desired output type Long`
- `score.py`: SacreBLEU rejected file-object inputs

After the compatibility fixes, both requested checkpoints were successfully evaluated.

Sources:

- `reports/final_eval/bleu/01_bleu_checkpoint_best_summary.txt`
- `reports/final_eval/bleu/02_bleu_checkpoint_last_summary.txt`

| Checkpoint | SacreBLEU | 1-gram | 2-gram | 3-gram | 4-gram | BP | Ratio | Hyp len | Ref len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_best.pt` | 32.79 | 66.5 | 40.8 | 26.9 | 18.0 | 0.968 | 0.969 | 124,203 | 128,189 |
| `checkpoint_last.pt` | 32.90 | 66.9 | 41.2 | 27.2 | 18.3 | 0.960 | 0.961 | 123,213 | 128,189 |

Interpretation:

- `checkpoint_last.pt` is slightly higher than `checkpoint_best.pt` on SacreBLEU in the recorded test.
- The difference is small: +0.11 SacreBLEU.

## 6. SubTransformer Architecture Used For Profiling

Source: `reports/final_eval/latency/03_flops_summary.txt`

```text
encoder:
  encoder_embed_dim: 512
  encoder_layer_num: 6
  encoder_ffn_embed_dim: [1024, 1024, 1024, 1024, 1024, 1024]
  encoder_self_attention_heads: [4, 4, 4, 4, 4, 4]

decoder:
  decoder_embed_dim: 512
  decoder_layer_num: 4
  decoder_ffn_embed_dim: [1024, 2048, 2048, 1024]
  decoder_self_attention_heads: [4, 4, 4, 2]
  decoder_ende_attention_heads: [4, 4, 4, 4]
  decoder_arbitrary_ende_attn: [2, -1, -1, -1]
```

This is the predefined HAT IWSLT14 De-En SubTransformer config named `HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.yml`.

## 7. FLOPs And Parameters

Sources:

- `reports/final_eval/latency/03_flops_summary.txt`
- `reports/profile_50k/03_flops_iwslt_hat_subtransformer.log`

| Metric | Value |
| --- | ---: |
| SubTransformer non-embedding params | 27,333,632 |
| Embedding params | 3,395,584 |
| Estimated total params | 30,729,216 |
| Total FLOPs | 1,476,804,784 |
| Total FLOPs without last layer | 1,320,607,920 |
| Last layer FLOPs | 156,196,864 |

## 8. Latency

### 8.1 GPU Latency

Successful V100 GPU latency logs exist from earlier profiling runs.

Sources:

- `reports/profile_50k/01_latency_gpu_iwslt_hat_subtransformer.log`
- `reports/final_eval/latency/01_gpu_latency.log`

| Source | Encoder mean | Encoder std | Decoder mean | Decoder std | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reports/profile_50k/01_latency_gpu_iwslt_hat_subtransformer.log` | 4.1288 ms | 0.0108 ms | 84.0841 ms | 0.8704 ms | 88.2130 ms |
| `reports/final_eval/latency/01_gpu_latency.log` | not extracted in tail | not extracted in tail | 84.4480 ms | 1.0129 ms | 88.5603 ms |

The final-eval rerun in `reports/final_eval/latency/02_gpu_latency_iwslt_hat_subtransformer.log` failed because CUDA was unavailable in that session:

```text
RuntimeError: No CUDA GPUs are available
```

### 8.2 CPU Latency

Source: `reports/final_eval/latency/01_cpu_latency_summary.txt`

| Metric | Value |
| --- | ---: |
| Encoder latency mean | 9.3368 ms |
| Decoder latency mean | 183.9023 ms |
| Decoder latency std | 3.0972 ms |
| Overall latency | 193.2391 ms |
| Iterations | 300 |

### 8.3 CPU Memory

CPU memory / max RSS was not successfully measured.

Sources:

- `reports/profile_50k/02_latency_cpu_memory_iwslt_hat_subtransformer.log`
- `reports/profile_50k/04_bleu_memory_checkpoint_best.log`

Failure:

```text
bash: /usr/bin/time: No such file or directory
```

CPU latency was later collected without `/usr/bin/time`, but CPU max RSS remains missing.

## 9. GPU Monitor Results

Source: `reports/final_eval/gpu_monitor_numeric_summary.txt`

| Metric | Value |
| --- | ---: |
| Samples | 130 |
| Peak memory | 18,359 MiB |
| Average memory | 5,354.08 MiB |
| Peak GPU utilization | 99.0% |
| Average GPU utilization | 25.58% |
| Peak power draw | 289.91 W |
| Average power draw | 98.60 W |

Source log:

- `reports/final_eval/gpu_monitor_final_eval.csv`
- `reports/final_eval/gpu_monitor_peak_summary.txt`
- `reports/final_eval/gpu_monitor_numeric_summary.txt`

Note: `reports/final_eval/HAT_FINAL_EXPERIMENT_SUMMARY.md` currently has stale GPU monitor values marked as N/A. The numeric summary above is the newer measured result.

## 10. Latency Dataset, Predictor, And Evolutionary Search

### 10.1 Existing Official Dataset Files

The repository contains pre-existing latency dataset CSV files under `hardware-aware-transformers/latency_dataset/`, including:

| File | Status |
| --- | --- |
| `iwslt14deen_gpu_titanxp_all.csv` | Present |
| `wmt14ende_gpu_titanxp_all.csv` | Present |
| `wmt14ende_cpu_xeon_all.csv` | Present |
| `wmt14ende_cpu_raspberrypi_all.csv` | Present |
| `wmt14enfr_gpu_titanxp_all.csv` | Present |
| `wmt14enfr_cpu_xeon_all.csv` | Present |
| `wmt14enfr_cpu_raspberrypi_all.csv` | Present |
| `wmt19ende_gpu_titanxp_all.csv` | Present |

These are bundled/pre-existing files, not newly collected V100 measurements from the final run.

### 10.2 New V100 Latency Dataset Attempt

Sources:

- `reports/final_eval/search/latency_dataset_v100_small.log`
- `reports/final_eval/search/latency_dataset_v100_small.pkl`

| Item | Value |
| --- | --- |
| Status | Failed |
| Intended run | bounded small V100 latency dataset attempt |
| Output pkl | present but 0 bytes |
| Valid samples | 0 |
| Failure reason | CUDA/NVIDIA driver unavailable |
| Error | `RuntimeError: No CUDA GPUs are available` |

### 10.3 Latency Predictor

Source: `reports/final_eval/search/latency_predictor_v100_skipped.log`

| Item | Value |
| --- | --- |
| Status | Skipped |
| Reason | V100 latency dataset was not collected |
| Predictor checkpoint | Not produced |
| RMSE / MAE / correlation | N/A |

### 10.4 Evolutionary Search

Source: `reports/final_eval/search/evo_search_v100_skipped.log`

| Item | Value |
| --- | --- |
| Status | Skipped |
| Reason | No V100 latency predictor checkpoint was produced |
| Top-k candidates | Not produced |
| Predicted latency | N/A |
| Inherited validation loss | N/A |
| Candidate config paths | N/A |

## 11. Artifact Checklist

| Artifact | Status | Path |
| --- | --- | --- |
| Environment log | Present | `reports/final_eval/00_environment_check.log` |
| Checkpoint inventory | Present | `reports/final_eval/01_checkpoint_check.log` |
| 50K training log | Present | `reports/supertrain/05_super_full50000_fromscratch.log` |
| 50K training extracted summary | Present | `final_artifacts/50k_training_summary.txt` |
| BLEU checkpoint best log | Completed | `reports/final_eval/bleu/01_bleu_checkpoint_best.log` |
| BLEU checkpoint best summary | Completed | `reports/final_eval/bleu/01_bleu_checkpoint_best_summary.txt` |
| BLEU checkpoint last log | Completed | `reports/final_eval/bleu/02_bleu_checkpoint_last.log` |
| BLEU checkpoint last summary | Completed | `reports/final_eval/bleu/02_bleu_checkpoint_last_summary.txt` |
| CPU latency log | Completed | `reports/final_eval/latency/01_cpu_latency_iwslt_hat_subtransformer.log` |
| CPU latency summary | Completed | `reports/final_eval/latency/01_cpu_latency_summary.txt` |
| GPU latency log | Completed earlier | `reports/profile_50k/01_latency_gpu_iwslt_hat_subtransformer.log` |
| GPU latency final-eval rerun | Failed due no CUDA | `reports/final_eval/latency/02_gpu_latency_iwslt_hat_subtransformer.log` |
| FLOPs log | Completed | `reports/final_eval/latency/03_flops_iwslt_hat_subtransformer.log` |
| FLOPs summary | Completed | `reports/final_eval/latency/03_flops_summary.txt` |
| GPU monitor CSV | Present | `reports/final_eval/gpu_monitor_final_eval.csv` |
| GPU monitor numeric summary | Completed | `reports/final_eval/gpu_monitor_numeric_summary.txt` |
| Runtime compatibility patch diff | Present | `reports/final_eval/patches/runtime_compatibility_patch.diff` |
| V100 latency dataset attempt | Failed | `reports/final_eval/search/latency_dataset_v100_small.log` |
| Latency predictor status | Skipped | `reports/final_eval/search/latency_predictor_v100_skipped.log` |
| Evolutionary search status | Skipped | `reports/final_eval/search/evo_search_v100_skipped.log` |

## 12. Completion Status Summary

| Item | Status |
| --- | --- |
| 50K SuperTransformer from-scratch training | Completed |
| Checkpoint generation | Completed |
| BLEU / SacreBLEU for `checkpoint_best.pt` | Completed |
| BLEU / SacreBLEU for `checkpoint_last.pt` | Completed |
| SubTransformer FLOPs / params | Completed |
| GPU latency | Completed in earlier V100 profiling logs |
| CPU latency | Completed |
| GPU monitor peak memory / utilization / power | Completed |
| CPU memory / max RSS | Missing |
| New V100 latency dataset collection | Failed |
| Latency predictor training | Skipped |
| Evolutionary search | Skipped |
| Top-k candidate retraining | Not run |

## 13. Key Numbers For Report

The most important numbers to cite:

| Category | Result |
| --- | --- |
| 50K training completion | 50,000 updates, 28,627.2 sec, OOM 0 |
| SuperTransformer size | about 69.31M params |
| Final largest validation | loss 4.019, nll_loss 2.448, ppl 5.46 |
| Final smallest validation | loss 4.434, nll_loss 2.891, ppl 7.42 |
| `checkpoint_best.pt` SacreBLEU | 32.79 |
| `checkpoint_last.pt` SacreBLEU | 32.90 |
| Profiled SubTransformer size | about 30.73M params |
| Profiled SubTransformer FLOPs | 1.477G |
| GPU latency | about 88.21-88.56 ms |
| CPU latency | 193.24 ms |
| GPU peak memory | 18,359 MiB |
| Average GPU utilization | 25.58% |
| Average power draw | 98.60 W |

## 14. Remaining Missing Data

| Missing item | Why missing | Suggested next action |
| --- | --- | --- |
| CPU memory / max RSS | `/usr/bin/time` is not installed | Install GNU time or use another memory profiler |
| New V100 latency dataset | Final eval runtime could not access CUDA | Rerun on V100 node with working NVIDIA driver |
| Latency predictor metrics | Requires newly collected latency dataset | Train after V100 latency dataset is available |
| Evolutionary search top-k | Requires latency predictor checkpoint | Run after predictor is trained |
| Top-k candidate retraining | Search did not produce candidates | Retrain after search outputs top-k configs |

