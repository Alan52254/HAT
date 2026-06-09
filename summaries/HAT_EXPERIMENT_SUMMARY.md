# HAT SuperTransformer 實驗摘要

## 1. Training

| 指標 | 結果 |
| --- | ---: |
| 任務 | IWSLT14 De-En |
| SuperTransformer training | 50,000 updates completed |
| 訓練方式 | from-scratch |
| 訓練時間 | 28,627.2 sec ≈ 7.95 hr |
| GPU | Tesla V100-SXM2-32GB |
| Peak training memory | 約 25.7GB / 32GB |
| OOM | 0 |
| SuperTransformer params | 約 69.31M |

## 2. BLEU / SacreBLEU

| Checkpoint | SacreBLEU |
| --- | ---: |
| checkpoint_best.pt | 32.79 |
| checkpoint_last.pt | 32.90 |

## 3. Latency

| Hardware | Encoder latency | Decoder latency | Overall latency |
| --- | ---: | ---: | ---: |
| V100 GPU | 4.1123 ms | 84.4480 ms | 88.5603 ms |
| CPU | 約 9.34 ms | 183.90 ms | 193.24 ms |

## 4. FLOPs / Params

| 指標 | 結果 |
| --- | ---: |
| Total FLOPs | 1.48G |
| Non-embedding params | 27.33M |
| Embedding params | 3.40M |
| Total params | 30.73M |

## 5. Search Pipeline 補做結果

| 階段 | 狀態 |
| --- | --- |
| V100 latency dataset collection | Attempted, failed due CUDA unavailable |
| Fallback dataset | Used bundled TitanXP latency dataset |
| Latency predictor | Completed |
| Predictor RMSE | 12.6486 ms |
| Predictor MAE | 9.5925 ms |
| Predictor MAPD | 0.0801 |
| Predictor correlation | 0.9716 |
| Full evolutionary search | Attempted, CPU-only validation too slow |
| Predictor-only top-k configs | Completed |
| Top-k CPU latency / FLOPs / Params | Completed |
| Top-k GPU latency | Failed due CUDA unavailable |

## 6. Predictor-only Top-K Candidates

| Rank | Pred. latency | CPU latency | Params | FLOPs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.99 ms | 112.15 ms | 21.99M | 1.096G |
| 2 | 83.72 ms | 119.66 ms | 25.47M | 1.256G |
| 3 | 83.61 ms | 115.92 ms | 24.95M | 1.232G |
| 4 | 83.11 ms | 102.84 ms | 21.80M | 1.062G |
| 5 | 82.97 ms | 111.78 ms | 22.85M | 1.136G |

## 7. 結論

本次已完成 HAT IWSLT14 De-En SuperTransformer 50K from-scratch training，並取得 BLEU、CPU/GPU latency、FLOPs、Params 與硬體需求數據。結果顯示單張 V100 32GB 可支撐該任務完整訓練，且同一 SubTransformer 在 GPU 與 CPU 上 latency 差異明顯，支持 hardware-aware search 的必要性。

後續 search pipeline 已完成 fallback prototype，但尚未完成正式 V100 latency dataset collection 與 full evolutionary search。正式 HAT reproduction 仍需在 PyTorch 可穩定存取 V100 的環境下重新收集 latency dataset、訓練 V100 predictor、執行 evolutionary search，並對 top-k candidates 進行 validation / fine-tuning。
