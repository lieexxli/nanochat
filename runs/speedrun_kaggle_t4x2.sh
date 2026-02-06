#!/bin/bash

# 此脚本配置用于训练你自己的 GPT-2 级别的 LLM（预训练 + 微调）
# 适配 Kaggle T4 x 2 GPU 环境（每张 T4 有 16GB 显存）
# 注意：T4 算力较弱，完整训练时间较长，建议先用较少数据测试

# 启动方式：
# bash runs/speedrun_kaggle_t4x2.sh

# 默认中间产物目录
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python 环境设置（使用 pip，适合 Kaggle 环境）

# 安装项目依赖（Kaggle 已预装大部分包，只需安装缺失的）
pip install -q tiktoken wandb huggingface_hub datasets blobfile tqdm
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention 安装失败，将使用标准注意力"

# -----------------------------------------------------------------------------
# wandb 设置
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 初始化报告
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器 (Tokenizer)

# 下载数据集（T4 资源有限，大幅减少数据量）
# 使用 2 个分片用于训练分词器
python -m nanochat.dataset -n 2
# 后台下载更多分片用于预训练（约 30 个分片 ≈ 0.8B tokens）
python -m nanochat.dataset -n 30 &
DATASET_DOWNLOAD_PID=$!
# 训练分词器
python -m scripts.tok_train
# 评估分词器
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基础模型 (预训练)
echo "正在等待数据集下载完成..."
wait $DATASET_DOWNLOAD_PID

# T4 x 2 低配版：
# - nproc_per_node=2: 使用 2 张 GPU
# - depth=12: 最小模型，确保显存足够
# - device-batch-size=2: 最小 batch，防止 OOM
# - 移除 --fp8: T4 不支持 FP8
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=12 \
    --target-param-data-ratio=5 \
    --device-batch-size=2 \
    --run=$WANDB_RUN

# 评估模型
torchrun --standalone --nproc_per_node=2 -m scripts.base_eval -- --device-batch-size=2

# -----------------------------------------------------------------------------
# 有监督微调 SFT

# 下载身份对话数据
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 运行 SFT 并评估
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
    --device-batch-size=2 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# 生成报告
python -m nanochat.report generate

echo "=========================================="
echo "训练完成！"
echo "与模型对话: python -m scripts.chat_cli -p \"你好\""
echo "启动 WebUI: python -m scripts.chat_web"
echo "=========================================="
