#!/bin/bash

# FSDP 训练快速启动脚本

# 检查是否有 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "警告: 未检测到 NVIDIA GPU，将使用 CPU 训练（会很慢）"
    echo ""
fi

# 检查 GPU 数量
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$NUM_GPUS" -eq "0" ]; then
    echo "单 GPU 或 CPU 模式"
    python3 fsdp_fp32.py --epochs 2 --batch_size 16
elif [ "$NUM_GPUS" -eq "1" ]; then
    echo "单 GPU 模式"
    python3 fsdp_fp32.py --epochs 2 --batch_size 32
else
    echo "多 GPU 模式 (${NUM_GPUS} GPUs)"
    torchrun --nproc_per_node=${NUM_GPUS} fsdp_fp32.py \
        --epochs 2 \
        --batch_size 32 \
        --log_interval 5
fi
