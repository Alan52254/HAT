# HAT Final Experiment Summary

Generated: 2026-06-09 00:55 CST

## 1. Environment

| Item | Value |
| --- | --- |
| Host | y0ny9xctr1780243117444-2b2bl |
| Python | /home/u4290247/miniconda3/envs/hat/bin/python, Python 3.7.12 |
| PyTorch | 1.7.1 |
| PyTorch CUDA | 10.2 |
| CUDA visible to PyTorch | False |
| GPU visible to nvidia-smi | No, NVIDIA driver unavailable |
| Repo commit | 70e5a279d080670208249fdd98ed731fa9bcc466 |
| Checkpoint path | /home/u4290247/HAT/checkpoints/iwslt14_super_full50000_fromscratch |
| Evaluation config | configs/iwslt14.de-en/subtransformer/HAT_iwslt14deen_titanxp@137.8ms_bleu@34.7.yml |

Checkpoint inventory was recorded in `01_checkpoint_check.log`; the directory size is about 20G and contains `checkpoint_best.pt`, `checkpoint_last.pt`, and `checkpoint_46_50000.pt`.

## 2. Training Completion

| Item | Value |
| --- | --- |
| SuperTransformer training | 50K updates completed |
| Training time | 28,627.2 sec |
| Peak training GPU memory | about 25.7GB / 32GB |
| OOM | 0 |
| SuperTransformer estimated params | about 69.31M |
| Checkpoints generated | yes |

## 3. Compatibility Patches

Only runtime compatibility patches were applied or verified; no model architecture or checkpoint weights were changed.

| File | Patch |
| --- | --- |
| fairseq/modules/multihead_attention_super.py | verified existing `q *= self.scaling` to `q = q * self.scaling` patch at line 198 |
| fairseq/search.py | changed `torch.div(..., out=LongTensor)` to `torch.floor_divide(...)` |
| score.py | changed SacreBLEU scoring to pass stripped line lists instead of file handles |

Backups and diff are in `reports/final_eval/patches/`, including `runtime_compatibility_patch.diff`.

## 4. BLEU Results

| Checkpoint | BLEU / SacreBLEU | Notes |
| --- | ---: | --- |
| checkpoint_best.pt | N/A | Attempt started, loaded test data/checkpoint, then stalled during CPU/no-CUDA generation; process was stopped and log tail saved. |
| checkpoint_last.pt | N/A | Skipped after `checkpoint_best.pt` stalled under the same CUDA-unavailable runtime. |

BLEU remains a required formal evaluation item once the V100 driver is visible again.

## 5. Latency Results

| Hardware | Encoder latency | Decoder latency | Overall latency | Notes |
| --- | ---: | ---: | ---: | --- |
| V100 GPU | N/A | N/A | N/A | Failed: `RuntimeError: No CUDA GPUs are available`. Prior external run reported about 88.213 ms overall. |
| CPU | 9.3368 ms | 183.9023 ms | 193.2391 ms | Completed with `--latcpu`; 300 iterations. |

## 6. FLOPs / Params

| Metric | Value |
| --- | ---: |
| FLOPs | 1,476,804,784 |
| FLOPs without last layer | 1,320,607,920 |
| Last layer FLOPs | 156,196,864 |
| Non-embedding params | 27,333,632 |
| Embedding params | 3,395,584 |
| Total estimated params | 30,729,216 |

These match the prior reported scale of about 1.48G FLOPs and about 30.73M total params.

## 7. GPU Memory / Utilization

| Metric | Value |
| --- | ---: |
| Samples | 0 |
| Peak memory | N/A |
| Avg memory | N/A |
| Peak utilization | N/A |
| Avg utilization | N/A |
| Peak power | N/A |
| Avg power | N/A |

GPU monitoring failed because `nvidia-smi` could not communicate with the NVIDIA driver.

## 8. Latency Dataset / Predictor

| Item | Status |
| --- | --- |
| V100 latency dataset | Failed |
| Sample count | 0 valid samples |
| Failure reason | `latency_dataset.py` reached `model.cuda()` and failed with `RuntimeError: No CUDA GPUs are available`. |
| Predictor training | Skipped |
| Predictor checkpoint | Not produced |
| RMSE / MAE / correlation | N/A |

The official IWSLT14 latency dataset config is `configs/iwslt14.de-en/latency_dataset/gpu_titanxp.yml`. A bounded 50-sample V100 attempt was logged to `search/latency_dataset_v100_small.log`.

## 9. Evolutionary Search

| Item | Status |
| --- | --- |
| Search | Skipped |
| Latency constraints | N/A |
| Top-k candidates | Not produced |
| Predicted latency | N/A |
| Inherited validation loss | N/A |
| Config paths | N/A |

Evolutionary search requires a usable V100 latency predictor checkpoint, which was not produced because V100 latency collection failed.

## 10. Conclusions

1. HAT SuperTransformer 50K from-scratch training has completed and checkpoints are present.
2. The prior TWCC V100 32GB run was sufficient for IWSLT14 SuperTransformer training.
3. SubTransformer CPU latency, FLOPs, and params were collected in this session.
4. BLEU remains a formal evaluation requirement; it was not completed here because the runtime cannot see CUDA and CPU generation stalled.
5. Changing device does not necessarily require retraining the SuperTransformer, but it does require collecting a new latency dataset and training a new latency predictor.
6. Expanding the search space or changing task usually requires retraining or a model-growth method such as LiGO / Net2Net.

## 11. Artifact Checklist

| Artifact | Status |
| --- | --- |
| 00_environment_check.log | present |
| 01_checkpoint_check.log | present |
| patches/runtime_compatibility_patch.diff | present |
| bleu/01_bleu_checkpoint_best.log | present, incomplete/stalled |
| bleu/02_bleu_checkpoint_last.log | present, skipped note |
| latency/01_cpu_latency_iwslt_hat_subtransformer.log | present |
| latency/02_gpu_latency_iwslt_hat_subtransformer.log | present, failed due no CUDA |
| latency/03_flops_iwslt_hat_subtransformer.log | present |
| gpu_monitor_final_eval.csv | present, contains nvidia-smi failure |
| gpu_monitor_numeric_summary.txt | present |
| latency_dataset_help.log | present |
| latency_predictor_help.log | present |
| evo_search_help.log | present |
