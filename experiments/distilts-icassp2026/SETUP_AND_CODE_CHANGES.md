# DistilTS 完整環境、資料、模型與程式修改紀錄

本文件記錄為了在 `/home/u4290247/DistilTS-ICASSP2026` 復現論文所做的所有
主要動作。原始碼基準為 commit
`0f6982a9606245747f82e5101884fdfb6e3ecafd`。

## 1. 主機與 GPU

實際在 sandbox 外執行 `nvidia-smi` 的結果：

- GPU：NVIDIA Tesla V100-SXM2 32GB
- Driver：535.161.08
- Driver 支援 CUDA：12.2
- 實驗使用 PyTorch CUDA wheel：CUDA 12.1
- Python：3.10

一般 sandbox 內看不到 `/dev/nvidia*`，因此在 sandbox 內執行
`nvidia-smi` 會誤報無法連到 driver。GPU 訓練必須在具有 GPU device 權限的環境執行。

## 2. 專案建立

```bash
git clone https://github.com/itsnotacie/DistilTS-ICASSP2026.git \
  /home/u4290247/DistilTS-ICASSP2026
cd /home/u4290247/DistilTS-ICASSP2026
```

因系統 Python 3.10 沒有 `ensurepip`，環境建立方式如下：

```bash
/usr/bin/python3.10 -m venv --without-pip .venv
/usr/bin/python3.10 -m pip --isolated --python .venv install pip setuptools wheel
```

## 3. PyTorch 與主要依賴

CUDA PyTorch：

```bash
.venv/bin/pip --isolated install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

其餘可重建環境的直接依賴已鎖定在 `requirements-reproduction.txt`：

```bash
.venv/bin/pip --isolated install -r requirements-reproduction.txt
```

重要相容性處理：

- README 最後指定 `accelerate==0.31.0`，但可取得的 Chronos 套件要求至少
  0.32，因此使用 `accelerate==0.32.1`。
- 使用 `chronos-forecasting==1.3.0`，以保留與 README 指定
  `transformers==4.40.1` 的相容性。
- 使用 `uni2ts==1.2.0`，對應程式中的
  `uni2ts.model.moirai.MoiraiModule/MoiraiForecast` API。
- Lightning 2.3.3 仍 import `pkg_resources`，因此 setuptools 固定 `<81`。
- CUDA wheel 安裝後會把 NumPy/fsspec 升級到不相容版本，必須再固定回
  `numpy==1.26.4`、`fsspec==2023.10.0`。
- `matplotlib` 是程式實際 import、但原 README 漏列的依賴。

完整傳遞依賴版本可隨時輸出：

```bash
.venv/bin/pip freeze | sort
```

目前 `pip check` 唯一警告來自 `reformer-pytorch` 新版傳遞套件要求
`einops>=0.8`，而 Uni2TS 1.2.0 明確固定 `einops==0.7.*`。實際使用到的
`LSHSelfAttention`、Uni2TS、三種 teacher adapter 都已完成 import 與推論測試。

## 4. 系統 library path 問題

主機的 `LD_LIBRARY_PATH` 含另一套系統 PyTorch library：

```text
/usr/local/lib/python3.10/dist-packages/torch/lib
```

它會讓 `.venv` 的 Python 載入錯誤的 `libc10.so`，造成 operator registry
錯誤。所有執行都必須移除該變數：

```bash
env -u LD_LIBRARY_PATH .venv/bin/python run.py ...
```

`run_local.sh` 已封裝這項處理：

```bash
./run_local.sh --task_name ...
```

## 5. 資料集

資料來源是原 README 的 Time-Series-Library Google Drive：

```bash
.venv/bin/gdown --folder \
  'https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2' \
  -O dataset
```

總下載量約 2.5GB。主實驗使用的檔案已解壓為：

- `dataset/ETT-small/ETTh1.csv`
- `dataset/ETT-small/ETTh2.csv`
- `dataset/ETT-small/ETTm1.csv`
- `dataset/ETT-small/ETTm2.csv`
- `dataset/weather/weather.csv`

原始 zip 仍保留在 `dataset/` 的各任務子目錄。

## 6. 預訓練模型

使用 `hf download REPO --local-dir PATH` 下載：

| Hugging Face repository | 本機路徑 | 約略大小 |
|---|---|---:|
| `autogluon/chronos-bolt-base` | `chronos-bolt-base/` | 784MB |
| `autogluon/chronos-bolt-small` | `chronos-bolt-small/` | 183MB |
| `Maple728/TimeMoE-50M` | `TimeMoE-50M/` | 217MB |
| `Maple728/TimeMoE-200M` | `TimeMoE-200M/` | 865MB |
| `Salesforce/moirai-1.1-R-base` | `MOIRAI-base/` | 349MB |
| `Salesforce/moirai-1.1-R-small` | `MOIRAI-small/` | 53MB |
| `Salesforce/moirai-1.1-R-large` | `MOIRAI-large/` | 1.2GB |

原 scripts 使用 `../MODEL_NAME`，因此在 `/home/u4290247/` 建立了指回專案
模型目錄的相容 symlink。模型實體仍全部放在專案資料夾中。

三種 adapter 都已驗證：Chronos、TimeMoE、MOIRAI 均能輸出正確 shape，且
所有值 finite。PyTorch CUDA matrix smoke test 亦通過。

## 7. 修改過的原始程式

### `run.py`

1. 把錯字 `CUDA_VISIBLE_DEVICE` 修正為 `CUDA_VISIBLE_DEVICES`。
2. 新增 `--seed`，並設定 Python、NumPy、PyTorch CPU/CUDA seed。
3. 原始版本固定 seed 2025，重跑五次會是同一 seed；修改後 queue 使用
   2025、2026、2027、2028、2029。

### `exp/exp_basic.py`

原始版本只看 `args.use_gpu` 就強制建立 `cuda:0`，即使 CUDA 不可用也會
crash。現在只有 `torch.cuda.is_available()` 為真才選 CUDA；MPS 亦加入
真正 availability 判斷，其餘安全退回 CPU。

### `exp/exp_DistilTS.py`
### `exp/exp_DistilTS_TKD.py`
### `exp/exp_DistilTS_FDKD.py`

三個檔案的 `profile_model()` 原本無條件呼叫 CUDA memory/synchronize API，
CPU 執行會在完成 metrics 後 crash。現在透過：

```python
use_cuda = self.device.type == 'cuda' and torch.cuda.is_available()
```

只在 CUDA 裝置執行 CUDA profiling；CPU 的 GPU memory 欄位回報 0。

### `models/TimeMoe.py`

原始 adapter 即使 `vt_loss=0`，仍要求 `output_hidden_states=True`，並在每個
batch 把大型 teacher hidden state 搬回 CPU。這些資料沒有進入 loss，卻顯著
拖慢 DLinear 與無 FTA 的實驗。

現在只有 `vt_loss=1` 才保留 hidden states：

```python
self.return_hidden = bool(getattr(configs, 'vt_loss', 0))
generated = self._generate_for_channels(x_ctx, return_hidden=self.return_hidden)
```

已分別測試 `vt_loss=0` 和 `vt_loss=1`：前者不保留 hidden，後者仍正確輸出
FTA 所需 `[B*D, T, 384]` hidden tensor。此修改不改 forecasting output 或 loss。

## 8. 新增的程式與文件

- `run_local.sh`：安全移除 host `LD_LIBRARY_PATH` 並執行 `run.py`。
- `reproduce_supported.py`：可恢復的 300-job 論文主矩陣 queue，目前未使用。
- `reproduction_status.py`：300-job queue 的狀態工具。
- `reproduce_core.py`：目前執行中的 30-job 核心方法／消融 queue。
- `core_reproduction_status.py`：核心 queue 完成/失敗及 epoch/iteration 狀態。
- `requirements-reproduction.txt`：直接依賴鎖定。
- `REPRODUCTION.md`：快速使用說明與 CPU smoke command。
- `REPRODUCTION_REPORT.md`：正式復現報告；queue 完成後加入統計結果。
- `.gitignore`：排除環境、資料、模型、checkpoint、log 與結果等大型產物。

## 9. 論文設定與 queue

論文文字設定：

- 五個資料集：ETTh1、ETTh2、ETTm1、ETTm2、Weather
- horizons：96、192
- hidden dimension：512
- feed-forward dimension：2048
- DLinear：10 epochs
- iTransformer：4 epochs
- patience：3
- 五次重複

作者 scripts 與論文存在落差：部分 iTransformer 腳本使用 2 或 10 epochs，
部分 DLinear 腳本使用 `d_ff=1024`。`reproduce_supported.py` 選擇以論文文字
設定為準。

queue 矩陣：

```text
3 teachers × 2 students × 5 datasets × 2 horizons × 5 seeds = 300 jobs
```

teachers：TimeMoE-50M、MOIRAI-base、Chronos-Bolt-base。

執行：

```bash
env -u LD_LIBRARY_PATH .venv/bin/python -u reproduce_supported.py --repeats 5
```

查看狀態：

```bash
.venv/bin/python reproduction_status.py
tail -f reproduction_logs/<job>.log
```

完成/失敗紀錄寫入 `reproduction_logs/manifest.csv`。只有 log 包含最終 profiling
summary 才視為完成；重啟 queue 時完成的 job 會跳過，失敗 job 會重跑。

核心 queue 採更嚴格驗收：process return code 必須為 0，log 必須同時包含最終
`mse/mae` 與 `Model Profiling Summary`，且 `checkpoints/` 下必須存在該 job 的
`checkpoint.pth`，才會在 manifest 標成 complete。

teacher inference 的 `gen_chunk` 設為 128。它只控制 frozen teacher 的推論切塊，
不改 student training batch、loss 或 optimizer；V100 實測約使用 4.7GB，仍低於
32GB。若個別長序列 job OOM，可把該值降為 64。

## 10. 已知無法由此 repo 完整重跑的項目

論文 Table 1 還包含 TimesFM、MOMENT 與 Chronos-large，但作者 repo 沒有：

- TimesFM adapter、下載或 scripts
- MOMENT adapter、下載或 scripts
- Chronos-large 的 README 下載指令與 released script

因此正式報告會把這些欄位標示為「作者釋出程式不支援」，不會用不同
implementation 的結果冒充原 repo 復現。

## 11. 驗證與目前狀態

已完成的 CPU smoke result（不是論文正式結果）：

- ETTh1、Chronos-Bolt-small → DLinear
- MSE 0.7269368
- MAE 0.5675045
- RMSE 0.8526059

原 300-job queue 因單張 V100 預估至少需 10–14 天，已停止並改為 30-job
核心復現。它優先驗證 Table 2 消融、Table 3 KD objective、Figure 3 teacher
與 scale。即時狀態請使用：

```bash
.venv/bin/python core_reproduction_status.py
```

## 12. 查看精確差異

所有針對原始 tracked source 的修改可用下列命令逐行審核：

```bash
git diff -- run.py exp/exp_basic.py exp/exp_DistilTS.py \
  exp/exp_DistilTS_TKD.py exp/exp_DistilTS_FDKD.py models/TimeMoe.py
```

新增檔案可用：

```bash
git status --short
```
