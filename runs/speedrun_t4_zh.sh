#!/bin/bash

# 这是一个榨干 16GB T4 显存，同时保证在 2 小时左右跑完的优化版脚本
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DTYPE=float16
mkdir -p $NANOCHAT_BASE_DIR

# --- Python venv ---
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

# --- 数据与分词 (Tokenizer) ---
# 下载 8 个切片训分词器，并在后台预先下载 15 个切片喂给稍后变大的模型
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 15 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

# --- 基础预训练 (Base Pretrain) ---
echo "正在等待数据集下载完成..."
wait $DATASET_DOWNLOAD_PID

# [火力全开] 显存利用率飙升、速度与时间的平衡配置
python -m scripts.base_train \
    --depth=8 \
    --target-param-data-ratio=4.5 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --max-seq-len=1024 \
    --window-pattern=L \
    --eval-every=1000 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run=$WANDB_RUN

python -m scripts.base_eval --device-batch-size=16 --max-per-task=100

# --- SFT 对话微调 ---
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --max-seq-len=1024 \
    --mmlu-epochs=1 \
    --gsm8k-epochs=1 \
    --eval-every=1000 \
    --chatcore-every=-1 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i sft --batch-size=16 --max-problems=100
python -m nanochat.report generate
