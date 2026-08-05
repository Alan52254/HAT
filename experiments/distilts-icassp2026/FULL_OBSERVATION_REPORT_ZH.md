# DistilTS 論文方法、模型架構、V100 階段性復現與資源觀察報告

> 報告日期：2026-08-05（Asia/Taipei）
>
> 論文：*Distilling Time Series Foundation Models for Efficient Forecasting*
>
> 論文版本：[arXiv:2601.12785v1](https://arxiv.org/abs/2601.12785)
>
> 原始程式：[itsnotacie/DistilTS-ICASSP2026](https://github.com/itsnotacie/DistilTS-ICASSP2026)
>
> 本次硬體：單張 NVIDIA Tesla V100-SXM2 32GB
>
> 本次狀態：30 組核心 queue 中完成 14 組、失敗 0 組，第 15 組由使用者主動停止

---

## 1. 執行摘要

DistilTS 的研究問題是：大型 Time Series Foundation Model（TSFM）雖有良好
zero-shot forecasting 能力，但參數量、推論延遲與部署成本過高；因此作者希望把
大型 teacher 的預測與隱表示知識，轉移到 DLinear 或 iTransformer 等小型
student。論文認為時間序列蒸餾有兩個特殊困難：

1. **預測步難度不均（task difficulty discrepancy）**：較近的 horizon 通常較容易，
   均勻平均 loss 容易讓短期步驟主導梯度，遠期步驟學得不足。
2. **teacher/student 架構不一致（architecture discrepancy）**：TSFM 常保留
   point-wise 或 patch-wise temporal states；iTransformer 則把每個變數壓成一個
   variate-wise token，兩者 hidden state 形狀無法直接對齊。

作者提出兩個模組：

- **Horizon-Weighted KD（HW）**：對越遠的預測步給越大權重。
- **Factorized Temporal Alignment（FTA）**：把 student 的 variate-wise embedding
  透過變數投影、時間 embedding 和輸出投影，重建 teacher 的 point-wise hidden
  representation。

本次原先估計完整矩陣需要 300 jobs（3 teachers × 2 students × 5 datasets ×
2 horizons × 5 seeds），單張 V100 至少需 10–14 天，且最慢 teacher 可能更久，
因此先縮成 30 組核心 queue，目標是重現 Table 2 消融、Table 3 KD objective
比較與 Figure 3 teacher/scale 比較。

停止時的實際成果如下：

- 14/30 formal jobs 完成，0 failed；第 15 組在 epoch 2、iteration 700 時停止。
- 已完成的 14 組全部屬於 **Table 2 / iTransformer / TimeMoE-50M**。
- ETTh1 的 2 horizons × 4 variants 已完整跑完。
- ETTm2 horizon 96 的 4 variants 已完整跑完。
- ETTm2 horizon 192 完成 Baseline 與 Only HW；Only FTA 中止、Full 未開始。
- Table 3 的 DLinear KD objective 比較尚未正式執行。
- Figure 3 的 teacher architecture/scale 比較尚未正式執行。
- 所有正式結果只有 seed 2025，不能寫成 mean ± std，也不能宣稱完成論文的
  五次重複。

最重要的稽核結論是：**目前釋出程式的實際 loss path 與論文方法存在關鍵落差**。
`horizon_weight_tau` 沒有真正傳入 `_weighted_mse`，而 FTA aligner 的參數沒有加入
optimizer，也沒有存入 checkpoint。這使得本次的 `Only FTA` 與 `Full DistilTS`
結果逐位完全相同。因此本次結果可以證明環境、資料、teacher/student pipeline、
uniform output KD 與 FTA-related gradient path 可執行，但**不能視為已正確驗證論文
所描述的 learnable FTA 與 horizon-weighted objective**。

---

## 2. 論文方法論

### 2.1 Teacher–student 總體關係

訓練時的資料流可表示為：

```text
歷史時間序列 x
   ├─> frozen TSFM teacher ──> teacher forecast y_teacher
   │                         └─> point-wise hidden H_teacher
   │
   └─> lightweight student ─> student forecast y_student
                              └─> variate-wise hidden H_student

ground truth y ──────────────> supervised MSE
y_teacher + y_student ───────> horizon-weighted KD loss
H_teacher + H_student ───────> FTA hidden alignment loss
```

Teacher 在蒸餾期間固定為 `eval()`，所有 teacher parameters 的
`requires_grad=False`。只有 student（以及論文理想設計中的 aligner）應被更新。
部署時不再需要 teacher，只保留小型 student，因此蒸餾成本是在 training time
支付，換取 deployment-time 的模型尺寸與推論速度。

### 2.2 Supervised objective

令 batch、channel、prediction length 分別為 \(B,C,T\)，ground truth 為 \(y\)，
student prediction 為 \(\hat y\)。論文 supervised loss 是一般 MSE：

\[
\mathcal{L}_{sup}=\frac{1}{BCT}\sum_{b,c,t}(y_{b,t,c}-\hat y_{b,t,c})^2.
\]

### 2.3 Horizon-Weighted Knowledge Distillation

論文使用隨 horizon 指數增加、再正規化到平均值為 1 的權重：

\[
w_t=\frac{\exp(\tau t/(T-1))}
{\frac{1}{T}\sum_{j=0}^{T-1}\exp(\tau j/(T-1))}.
\]

KD loss 為：

\[
\mathcal{L}_{KD}=\frac{1}{BCT}\sum_{b,c,t}
w_t(\hat y^T_{b,t,c}-\hat y_{b,t,c})^2.
\]

\(\tau=0\) 時退化為 uniform output KD；\(\tau>0\) 時越遠的 horizon 權重越大。
本次 queue 設定的 `kd_alpha=0.5`，程式總 loss 寫成：

```text
L = L_sup + 0.5 * L_KD
```

這不是 `(1-alpha)*L_sup + alpha*L_KD`，而是保留完整 supervised loss，再加上
KD regularizer。

### 2.4 Factorized Temporal Alignment（FTA）

論文 teacher hidden state 為：

\[
H^T \in \mathbb{R}^{B\times D\times T\times d_T},
\]

student iTransformer 每個變數只有一個 embedding：

\[
H^S \in \mathbb{R}^{B\times D\times d_S}.
\]

FTA 先把 student embedding 投影到 latent dimension \(u\)，再乘上可學習的
time embedding，最後映射到 teacher hidden dimension：

\[
\hat H^T_{b,d,t}=W_{out}\,\phi((W_s h^S_{b,d})\odot E_t).
\]

對齊 loss 為：

\[
\mathcal{L}_{FTA}=\frac{1}{BDT}\sum_{b,d,t}
\|\hat H^T_{b,d,t}-H^T_{b,d,t}\|^2.
\]

repo 中 `VarTimeFactorAligner` 的預設 latent dimension 是 256，student dimension
是 512，TimeMoE teacher hidden dimension 是 384，非線性是 GELU。為控制顯存，
時間軸以 `block_T=128` 分塊計算。程式把 FTA loss 乘 0.3：

```text
L = L_sup + 0.5 * L_KD + 0.3 * L_FTA
```

### 2.5 論文比較的其他 KD objective

- **T-KD（Trend Projection KD）**：先對 student/teacher forecast 做 trend
  projection，再以 MSE 對齊粗粒度趨勢。
- **FD-KD（Frequency & Difference KD）**：同時對齊 rFFT log-magnitude 與
  一階 temporal difference，以頻域與局部變化共同蒸餾。

這兩個 objective 原定在核心 queue 的 Table 3 執行，但停止前尚未開始，因此
本次不能比較 DistilTS、FD-KD 與 T-KD 的實測優劣。

---

## 3. 模型架構與模型之間的關係

### 3.1 本次實際 teacher：TimeMoE-50M

本次所有 14 組正式結果都使用 TimeMoE-50M teacher。本機模型約 217MB，adapter
把 multivariate input `[B,L,D]` 轉成 `[B*D,L]`，讓每個 channel 各自進入
TimeMoE autoregressive generation，再 reshape 回 `[B,pred_len,D]`。

重要實作關係：

- teacher 為 frozen pretrained causal TSFM，只提供 forecast/hidden supervision。
- `gen_chunk=128` 控制 teacher 一次生成多少 channel-series，目的是降低顯存，
  不改 student batch size 或 loss 定義。
- `vt_loss=0` 時不要求 teacher hidden state，可減少 CPU copy 與生成負擔。
- `vt_loss=1` 時保存 TimeMoE prefill hidden state，shape 約為
  `[B*D, seq_len, 384]`，供 FTA 使用。

### 3.2 本次實際 student：iTransformer

iTransformer 採 inverted/variate-wise tokenization：每個變數是一個 token，
時間維度被 embedding 壓入 token feature。這比 point-wise tokenization 的 token
數少，適合作為輕量 student。本次設定：

| 項目 | 設定 |
|---|---:|
| encoder layers | 1 |
| attention heads | 4 |
| hidden dimension | 512 |
| feed-forward dimension | 2048 |
| dropout | 0.1 |
| activation | GELU |
| batch size | 32 |
| learning rate | 1e-4 |
| optimizer | Adam |
| train epochs | 4 |
| early stopping patience | 3 |
| seed | 2025 |

模型會先對 input 做 per-series mean/std normalization，經 inverted embedding 與
Transformer encoder，再由 linear projection 直接產生 prediction horizon，最後反
正規化。

prediction length 會改變最後 linear head，因此參數量不同：

| Prediction length | Student parameters | 平均單次 student forward |
|---:|---:|---:|
| 96 | 3,465,312 | 0.001748 s |
| 192 | 3,776,704 | 0.001749 s |

這裡的 inference time 是 profiler 對一個 student batch 的單次 forward，不是完整
test set latency，也不是 teacher inference。論文宣稱最高 1/150 parameters、6000×
加速；本次沒有用相同 RTX 3090 與完整 teacher profiling protocol，因此沒有獨立
驗證該倍數。

### 3.3 原定但未正式完成的 student/teachers

- **DLinear student**：先用 moving average 分出 seasonal/trend，分別做線性
  projection 後相加。原定 10 epochs，用於 Table 3 與 Figure 3。
- **TimeMoE-200M**：原定與 50M 比較 teacher scale。
- **Chronos-Bolt-base**：原定比較另一種 teacher architecture。
- **MOIRAI-base**：原定比較另一種 teacher architecture。

Chronos、TimeMoE、MOIRAI 三種 adapter 都做過 finite output/shape smoke test；但
只有 TimeMoE-50M → iTransformer 進入本次正式結果。另有一組
Chronos-Bolt-small → DLinear 的 CPU one-epoch smoke result（MSE 0.7269368、
MAE 0.5675045、RMSE 0.8526059），只證明 pipeline 能跑，不能當論文正式結果。

---

## 4. 實驗環境、資料與相容性修改

| 項目 | 本次環境 | 論文環境 |
|---|---|---|
| GPU | Tesla V100-SXM2 32GB | RTX 3090 24GB |
| Driver | 535.161.08 | 未在論文詳列 |
| CUDA compatibility | 12.2（driver） | 未在論文詳列 |
| PyTorch | 2.4.1+cu121 | 未在論文詳列 |
| Python | 3.10 | 未在論文詳列 |
| repetitions | 1 seed（2025） | 5 次平均 |

正式使用資料：

- ETTh1，7 channels；p96 使用 lookback 512，p192 使用 lookback 1024。
- ETTm2，7 channels；p96 使用 lookback 512，p192 使用 lookback 1024。
- z-score normalization；training objective 是 MSE。

為使 repo 可在 V100 container 執行，本次做過：

1. 修正 `CUDA_VISIBLE_DEVICES` 拼字與 CPU/GPU device fallback。
2. 對 CPU path 加上 CUDA profiling guards。
3. 新增真正可切換的 `--seed`，避免五次執行仍固定 seed 2025。
4. `vt_loss=0` 時避免不使用的 TimeMoE hidden-state CPU copies。
5. 執行時移除 host `LD_LIBRARY_PATH`，避免載入另一套 PyTorch `libc10.so`。
6. 固定 Chronos/Accelerate、Uni2TS、NumPy、fsspec 等相容版本。
7. 建立 resumable queue、manifest acceptance checks、status monitor 與資源監控器。

完整套件與程式差異分別保存在 `pip_freeze.txt` 與
`local_code_changes.patch`。

---

## 5. 原定實驗矩陣與實際完成進度

### 5.1 原始完整構想

```text
3 teachers × 2 students × 5 datasets × 2 horizons × 5 seeds = 300 jobs
```

因單張 V100 預估至少需要 10–14 天，改成 30-job core queue：

| 區塊 | 目的 | Jobs | 停止時完成 |
|---|---|---:|---:|
| Table 2 | Baseline/HW/FTA/Full 消融 | 16 | 14 complete、1 interrupted、1 not started |
| Table 3 | DistilTS-L/FD-KD/T-KD | 6 | 0 |
| Figure 3 | teacher architecture/scale | 8 | 0 |
| **總計** |  | **30** | **14 complete** |

### 5.2 各資料集完成狀態

| Dataset | Horizon | Baseline | Only HW | Only FTA | Full |
|---|---:|---|---|---|---|
| ETTh1 | 96 | complete | complete | complete | complete |
| ETTh1 | 192 | complete | complete | complete | complete |
| ETTm2 | 96 | complete | complete | complete | complete |
| ETTm2 | 192 | complete | complete | interrupted | not started |

中止 job `Table2_OnlyFTA_ETTm2_p192` 已完成 epoch 1，最後確認到 epoch 2、
iteration 700/1043。它沒有 final test metrics 與 profiling，故未計入 formal result。

---

## 6. 階段性實驗結果

### 6.1 全部正式結果

| Dataset | Pred | Variant | MSE | MAE | RMSE |
|---|---:|---|---:|---:|---:|
| ETTh1 | 96 | Baseline | 0.385929 | 0.412591 | 0.621231 |
| ETTh1 | 96 | Only HW label | 0.382014 | 0.408544 | 0.618073 |
| ETTh1 | 96 | Only FTA label | 0.373251 | 0.401875 | 0.610943 |
| ETTh1 | 96 | Full label | 0.373251 | 0.401875 | 0.610943 |
| ETTh1 | 192 | Baseline | 0.426974 | 0.444530 | 0.653433 |
| ETTh1 | 192 | Only HW label | 0.414808 | 0.435143 | 0.644056 |
| ETTh1 | 192 | Only FTA label | 0.413779 | 0.434307 | 0.643256 |
| ETTh1 | 192 | Full label | 0.413779 | 0.434307 | 0.643256 |
| ETTm2 | 96 | Baseline | 0.179390 | 0.268338 | 0.423544 |
| ETTm2 | 96 | Only HW label | 0.173252 | 0.262987 | 0.416235 |
| ETTm2 | 96 | Only FTA label | 0.171546 | 0.262293 | 0.414181 |
| ETTm2 | 96 | Full label | 0.171546 | 0.262293 | 0.414181 |
| ETTm2 | 192 | Baseline | 0.249203 | 0.320550 | 0.499203 |
| ETTm2 | 192 | Only HW label | 0.238954 | 0.313034 | 0.488829 |

表中的 `label` 特別提醒：名稱是 queue configuration 名稱，不等於已證明程式
真正實作相對應論文模組；第 8 節會說明原因。

### 6.2 相對 Baseline 的表面改善

| Dataset | Pred | Only HW label | Only FTA label | Full label |
|---|---:|---:|---:|---:|
| ETTh1 | 96 | 1.01% | 3.28% | 3.28% |
| ETTh1 | 192 | 2.85% | 3.09% | 3.09% |
| ETTm2 | 96 | 3.42% | 4.37% | 4.37% |
| ETTm2 | 192 | 4.11% | incomplete | not started |

ETTh1 兩 horizons 平均 MSE：

| Variant | 本次平均 MSE | 論文 Table 2 平均 MSE |
|---|---:|---:|
| Baseline | 0.406451 | 0.414 |
| Only HW label | 0.398411 | 0.402 |
| Only FTA label | 0.393515 | 0.399 |
| Full label | 0.393515 | 0.395 |

### 6.3 與論文 Table 2 的逐格差異

`delta = 本次 MSE - 論文 MSE`，負值代表本次單一 seed 數字較低。

| Dataset | Pred | Variant | 本次 | 論文 | Delta |
|---|---:|---|---:|---:|---:|
| ETTh1 | 96 | Baseline | 0.385929 | 0.386 | -0.000071 |
| ETTh1 | 96 | Only HW label | 0.382014 | 0.382 | +0.000014 |
| ETTh1 | 96 | Only FTA label | 0.373251 | 0.378 | -0.004749 |
| ETTh1 | 96 | Full label | 0.373251 | 0.374 | -0.000749 |
| ETTh1 | 192 | Baseline | 0.426974 | 0.441 | -0.014026 |
| ETTh1 | 192 | Only HW label | 0.414808 | 0.421 | -0.006192 |
| ETTh1 | 192 | Only FTA label | 0.413779 | 0.419 | -0.005221 |
| ETTh1 | 192 | Full label | 0.413779 | 0.415 | -0.001221 |
| ETTm2 | 96 | Baseline | 0.179390 | 0.180 | -0.000610 |
| ETTm2 | 96 | Only HW label | 0.173252 | 0.176 | -0.002748 |
| ETTm2 | 96 | Only FTA label | 0.171546 | 0.174 | -0.002454 |
| ETTm2 | 96 | Full label | 0.171546 | 0.172 | -0.000454 |
| ETTm2 | 192 | Baseline | 0.249203 | 0.250 | -0.000797 |
| ETTm2 | 192 | Only HW label | 0.238954 | 0.246 | -0.007046 |

數值相近表示資料切分、模型、teacher 與大部分超參數很可能對齊；但本次只有
seed 2025，論文是五次平均，而且存在第 8 節的實作問題，因此不能用「誤差很小」
直接宣稱完整復現成功。

---

## 7. 訓練時間、epochs 與 iterations

### 7.1 每個 job 的執行量

| Job | Steps/epoch | Epochs | 完整 train iterations | Epoch-loop time | Manifest wall time |
|---|---:|---:|---:|---:|---:|
| Baseline ETTh1 p96 | 252 | 4 | 1,008 | 4.27 min | 14.85 min |
| Only HW ETTh1 p96 | 252 | 4 | 1,008 | 44.13 min | 54.85 min |
| Only FTA ETTh1 p96 | 252 | 4 | 1,008 | 47.06 min | 57.63 min |
| Full ETTh1 p96 | 252 | 4 | 1,008 | 46.48 min | 56.93 min |
| Baseline ETTh1 p192 | 233 | 4 | 932 | 4.21 min | 14.58 min |
| Only HW ETTh1 p192 | 233 | 4 | 932 | 96.39 min | 未寫入 manifest |
| Only FTA ETTh1 p192 | 233 | 4 | 932 | 103.80 min | 114.64 min |
| Full ETTh1 p192 | 233 | 4 | 932 | 103.46 min | 113.92 min |
| Baseline ETTm2 p96 | 1,062 | 4 | 4,248 | 4.71 min | 15.32 min |
| Only HW ETTm2 p96 | 1,062 | 4 | 4,248 | 170.49 min | 181.21 min |
| Only FTA ETTm2 p96 | 1,062 | 4 | 4,248 | 182.84 min | 193.46 min |
| Full ETTm2 p96 | 1,062 | 4 | 4,248 | 184.26 min | 195.03 min |
| Baseline ETTm2 p192 | 1,043 | 4 | 4,172 | 4.89 min | 15.80 min |
| Only HW ETTm2 p192 | 1,043 | 4 | 4,172 | 420.32 min | 431.20 min |
| Only FTA ETTm2 p192 | 1,043 | 1 完整 + 部分 epoch 2 | 至少 1,743 | epoch 1 為 112.39 min | interrupted |

`Epoch-loop time` 只包含 training loop。`Manifest wall time` 還包含資料載入、每個
epoch 的 validation/test、最終 test、checkpoint 與 profiling，所以兩者不相同。

彙總：

- 14 個 complete jobs：33,096 個完整 training iterations。
- 中止 job：另確認至少 1,743 iterations（完整 epoch 1 的 1,043，加上 epoch 2
  最後輸出的 700；實際可能多於 700、但少於下一個 100-step log）。
- 14 個 complete jobs 的 epoch training loop 合計 85,037.6 秒，即 23.62 小時。
- manifest 有精確 wall time 的 13 個 complete jobs 合計 87,564.5 秒，即
  24.32 小時；`OnlyHW_ETTh1_p192` 因舊 runner 在寫 manifest 前中止，只有完整
  log/checkpoint，沒有 wall-time row。
- queue 曾停止並續跑，因此「日曆經過時間」不能簡單等同以上任一數值。

### 7.2 每 iteration 的觀察

由 epoch time / steps 粗估：

- Baseline student-only：約 0.06–0.27 秒/iteration。
- TimeMoE KD、p96：約 2.4–2.8 秒/iteration。
- TimeMoE KD、p192：約 6.2–6.7 秒/iteration。
- FTA 需要 teacher hidden states 與 block-wise alignment，通常比 output-only KD
  再慢一些。

時間主要花在 frozen TimeMoE autoregressive generation，而不是 3–4M parameter
iTransformer student。prediction length 從 96 變成 192 時，teacher 要生成更多
步，而且 lookback 從 512 增為 1024，因此 ETTm2 p192 output KD 一組約需 7.2
小時 wall time。

---

## 8. 資源使用量

資源監控器每 10 秒記錄一次，從 2026-08-04 08:24:39 到
2026-08-05 08:21:18，共 8,584 rows、23.94 小時。監控從 queue 第 7 組附近才
開始，因此**不包含最前面的約 6 組**，下表只代表被監控區間。

| 指標 | 平均 | Median | P95 | Max |
|---|---:|---:|---:|---:|
| GPU utilization | 86.9% | 97% | 100% | 100% |
| GPU memory | 10,948.7MB | 13,333MB | 20,055MB | 20,639MB |
| GPU power | 228.2W | 252.1W | 303.5W | 324.2W |
| GPU temperature | 35.9°C | 37°C | 38°C | 38°C |
| Experiment process CPU | 97.3% | 97.7% | 106.0% | 108.0% |
| Experiment process RSS | 1,821.5MB | 1,813.7MB | 2,177.3MB | 2,633.4MB |

以每筆 power sample 對下一個 10 秒區間做矩形積分，被監控區間的 GPU 能耗估計：

\[
E \approx 5.47\text{ kWh}.
\]

限制：

- 5.47 kWh 只算 `nvidia-smi` GPU board power，不含 CPU、RAM、storage、風扇、
  datacenter PUE，也不含監控啟動前的前 6 組。
- CPU percent 採 `ps` 語意，約 100% 代表大約一個 CPU core 的總量，不是整台
  多核心節點滿載。
- 系統 RAM 平均約 98GB 是共享節點的 system-wide 值，不能全部歸因於本實驗；
  較可信的 experiment-process RSS 平均約 1.82GB、峰值約 2.63GB。
- 共享檔案系統的 disk used/free 會受其他使用者影響，因此不把其變化當作本實驗
  寫入量。

Profiler 顯示 student-only forward peak memory 約 53–57MB；蒸餾 job profiler
觀察到約 275–283MB。但這是最終 student profiling code 的記憶體口徑，不等於
整個 training process。整體 training 的 `nvidia-smi` 峰值 20.64GB 才包含 teacher
generation、hidden states、student、activations 和 CUDA allocator。

---

## 9. 程式碼稽核與結果解釋的關鍵限制

### 9.1 Horizon weights 沒有接入實際 KD loss

程式有正確定義 `_horizon_weights()`，但 training loop 實際呼叫：

```python
kd_loss = self._weighted_mse(outputs, t_outputs.float())
```

沒有建立或傳入 `w_t`。因此 `horizon_weight_tau=0` 與 `1` 對實際 KD loss 沒有
影響。現有 queue 中：

- `Only HW` 實際上是 **uniform output KD**，不是論文的 horizon-weighted KD。
- `Full` 的 horizon weighting 部分實際未啟用。

### 9.2 FTA aligner 沒有加入 optimizer

`_select_optimizer()` 只有：

```python
optim.Adam(self.model.parameters(), ...)
```

`self.vt_aligner.parameters()` 沒有被加入 optimizer。backward 時 FTA loss 仍會把
梯度傳回 student hidden state，但 `W_s`、time embedding 和 `W_out` 本身不會被
更新，等同使用固定的隨機映射，而不是論文描述的 learnable alignment module。

此外 checkpoint 只保存 `self.model.state_dict()`，未保存 aligner；profiler 的
parameter count 也只統計 student model，沒有計入 aligner。

### 9.3 Logged train loss 不含 FTA loss

training loop 先 `train_loss.append(loss.item())`，之後才：

```python
loss += 0.3 * vt_loss
```

所以 FTA job log 顯示的 `Train Loss` 是 supervised + output KD，不是 backward
真正使用的 total loss。validation/test loss 則是純 supervised MSE，這一點合理，
但報告不能拿 logged train loss 評估 FTA loss 是否下降。

### 9.4 Only FTA 與 Full 結果完全相同不是巧合

三個已完成條件中，Only FTA 與 Full 的 MSE/MAE/RMSE、各 epoch train/validation/
test loss 都逐位相同。因 seed、資料順序與其餘設定相同，而 `tau` 沒進 loss，
這兩組實際計算圖相同，所以得到 deterministic identical result。這是程式稽核與
實驗數據互相印證的證據。

### 9.5 可合理下的結論

本次可以支持：

- V100 上的 data → TimeMoE teacher → iTransformer student pipeline 可完整執行。
- student-only baseline 與 frozen-teacher output KD 可產生穩定、finite 結果。
- 現有 FTA-related loss path 對 student 梯度的設定，表面上比 baseline 有較低
  MSE，但 aligner 本身沒有被訓練。
- 多數結果與論文 Table 2 同量級，資料/模型/超參數大致對齊。
- teacher generation 是 training 成本的主要瓶頸，並且 p192 顯著慢於 p96。

本次不能支持：

- 正確 horizon weighting 已被驗證。
- learnable FTA 已被驗證。
- Full DistilTS 比 Only FTA 更好。
- DistilTS 優於 FD-KD/T-KD。
- TimeMoE 比 Chronos/MOIRAI teacher 更適合，或 200M 比 50M 更好。
- 五 seeds 的統計顯著性。
- 論文完整矩陣或 6000× inference speedup 已被重現。

---

## 10. 與論文釋出內容的其他落差

1. 論文使用五次重複並報平均；原始 `run.py` 固定 seed 2025，本次才加入
   `--seed`，但目前只完成第一個 seed。
2. 論文文字指定 iTransformer 4 epochs、DLinear 10 epochs、d_ff=2048；部分
   released scripts 使用 2 或 10 iTransformer epochs、d_ff=1024。本 queue 採
   論文文字設定。
3. 論文 Table 1 包含 TimesFM、MOMENT、Chronos-large；repo 缺少 TimesFM/MOMENT
   adapter/scripts，README/released scripts 也沒有完整 Chronos-large path。
4. 論文 Table 1 報告 across-teacher 的 best distilled result；本次只正式跑
   TimeMoE-50M，不能做 best-teacher selection。
5. V100 與論文 RTX 3090 不同，wall time 與功耗不可直接比較。

---

## 11. 建議的正確後續實驗

在繼續消耗 GPU 前，應先修正兩個核心 implementation 問題：

1. 每個 batch 依 prediction length 建立 `w_t`，並傳給 `_weighted_mse`：

   ```python
   w_t = self._horizon_weights(outputs.size(1), outputs.device)
   kd_loss = self._weighted_mse(outputs, t_outputs.float(), w_t)
   ```

2. 把 student 與 FTA aligner 一起交給 optimizer，並把 aligner 納入 checkpoint：

   ```python
   parameters = list(self.model.parameters())
   if self.args.vt_loss:
       parameters += list(self.vt_aligner.parameters())
   optimizer = Adam(parameters, lr=...)
   ```

修正後最小驗證順序：

1. ETTh1 p96：Baseline、uniform KD、Only HW、Only FTA、Full，各跑相同 seed。
2. 檢查 `tau=0` 與 `tau=1` 的 KD loss/gradient 確實不同。
3. 記錄 aligner parameter norm 在 optimizer step 前後確實改變。
4. checkpoint reload 後確認 student + aligner state 一致。
5. 再補 ETTh1 p192 與 ETTm2 p96/p192。
6. 最後對關鍵 Baseline/Full 補 3–5 seeds；不要先把所有 30 組平均鋪開。

如果報告時程優先，目前最誠實的標題是：

> **DistilTS 核心方法分析、V100 階段性復現與 released-code 實作稽核**

---

## 12. 可直接提供給教授或另一個 ChatGPT 的摘要

本實驗研究 DistilTS 將大型時間序列基礎模型的知識蒸餾到輕量 DLinear 或
iTransformer student。論文的兩個核心方法是 horizon-weighted KD 與 Factorized
Temporal Alignment：前者提高遠期預測步的權重，後者把 iTransformer 的
variate-wise embedding 重建成 teacher 的 point-wise hidden states。完整論文使用
五個資料集、prediction lengths 96/192、五次重複與單張 RTX 3090。本次在單張
V100 32GB 上建立相容環境、下載資料與 TimeMoE/Chronos/MOIRAI checkpoints、修正
CUDA/seed/library 相容性，並把原 300-job 計畫縮成 30-job 核心 queue。

停止時完成 14/30 jobs、0 failed，均為 TimeMoE-50M teacher 蒸餾 iTransformer
的 Table 2 消融；ETTh1 兩 horizons 與 ETTm2 p96 完整，ETTm2 p192 完成 Baseline
與 Only HW，第 15 組在 epoch 2 iteration 700 主動停止。14 組共完成 33,096
training iterations，完整 epoch-loop 約 23.62 小時；資源監控涵蓋最後 23.94
小時，GPU 平均利用率 86.9%、峰值顯存 20.64GB、平均功耗 228.2W、估算 GPU
能耗 5.47kWh。student 約 3.47–3.78M parameters，單次 profiler forward 約
1.75ms。

結果與論文 Table 2 大致同量級，表面上 KD variants 比 baseline 低約 1.0–4.4%
MSE。然而原始碼稽核發現 horizon weights 沒有傳入實際 KD loss，FTA aligner 也
沒有加入 optimizer/checkpoint；因此 Only FTA 與 Full 在相同 seed 下得到逐位
完全相同的結果。這代表目前成果是可稽核的階段性 pipeline reproduction 與
released-code audit，而不是兩個論文核心模組的完整正確復現。正式報告應清楚
標示單一 seed、部分矩陣、硬體差異與上述實作落差。

---

## 13. 證據與可追溯檔案

- `results_snapshot.csv`：14 complete + 1 incomplete 的機器可讀 metrics。
- `core_reproduction_logs/*.log`：所有 epoch、iteration、loss、test metrics 與
  profiling 原始輸出。
- `core_reproduction_logs/manifest.csv`：job return code 與 wall time。
- `core_reproduction_logs/resource_usage.csv`：10 秒級 GPU/CPU/RAM/disk telemetry。
- `STOPPED_STATE_2026-08-05.md`：停止點與程序狀態。
- `SETUP_AND_CODE_CHANGES.md`：環境、資料、模型與相容性修改。
- `reproduce_core.py`：30-job queue 定義。
- `exp/exp_DistilTS.py`：loss、FTA 與 optimizer 的實際實作。
- `local_code_changes.patch`：相對原始 checkout 的程式修改。

所有本報告的本次實驗數字都可由以上 raw artifacts 重算；尚未執行的實驗不以
推測數值填補。
