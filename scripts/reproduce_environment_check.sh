#!/usr/bin/env bash
set -e

cd ~/HAT/hardware-aware-transformers

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hat

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "===== Python ====="
which python
python -V

echo "===== CUDA ====="
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("torch cuda=", torch.version.cuda)
print("cuda available=", torch.cuda.is_available())
print("device count=", torch.cuda.device_count())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
PY

echo "===== GPU ====="
nvidia-smi || true

echo "===== Checkpoints ====="
ls -lah ~/HAT/checkpoints/iwslt14_super_full50000_fromscratch || true

echo "===== Reports ====="
find ~/HAT/reports -maxdepth 3 -type f | sort | tail -50 || true
