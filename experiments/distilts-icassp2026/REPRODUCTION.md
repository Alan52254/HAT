# Local reproduction

This checkout contains the Python 3.10 environment, all datasets listed by the
upstream README, and all seven teacher checkpoints. Run commands from the
repository root through `./run_local.sh`; it isolates this project from the
host image's incompatible system PyTorch libraries.

The upstream dependency list is internally inconsistent: Chronos requires
`accelerate>=0.32`, while the README finishes by pinning `0.31.0`. This setup
uses the smallest compatible change, `accelerate==0.32.1`, together with
`chronos-forecasting==1.3.0`, which supports the pinned Transformers 4.40.1.

## Prepared paths

- Forecasting CSV files: `dataset/ETT-small/` and `dataset/weather/`
- Full downloaded dataset archive collection: `dataset/`
- Chronos: `chronos-bolt-base/`, `chronos-bolt-small/`
- TimeMoE: `TimeMoE-50M/`, `TimeMoE-200M/`
- MOIRAI: `MOIRAI-small/`, `MOIRAI-base/`, `MOIRAI-large/`

The original scripts refer to checkpoints one directory above the checkout.
Compatible symlinks have been created there. Run a complete upstream script
with the virtual environment first on `PATH` and the host library path removed:

```bash
env -u LD_LIBRARY_PATH PATH="$PWD/.venv/bin:$PATH" bash scripts/ChronosDistill/DLinear.sh
```

## Verified CPU smoke experiment

The following one-epoch DistilTS run was completed on ETTh1 with a
Chronos-Bolt teacher and DLinear student:

```bash
./run_local.sh \
  --task_name Exp_DistilTS --is_training 1 \
  --pretrained_path ./chronos-bolt-small \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id smoke_ETTh1_32_8_DLinear_Chronos \
  --model DLinear --data ETTh1 --features M \
  --seq_len 32 --label_len 0 --pred_len 8 \
  --d_model 32 --enc_in 7 --dec_in 7 --c_out 7 \
  --TSFModel Chronos --kd_alpha 0.5 --vt_loss 0 \
  --gen_chunk 100000 --batch_size 512 --num_workers 0 \
  --train_epochs 1 --patience 1 --itr 1
```

Observed test metrics: MSE 0.7269368, MAE 0.5675045, RMSE 0.8526059.
The checkpoint is retained under `checkpoints/`.
