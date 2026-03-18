#!/bin/bash

# 此脚本配置为训练您自己的小型 LLM (预训练 + 微调)
# 它旨在单张 T4 GPU (16GB 显存) 上运行，大约需要 2 小时完成。
#
# 相较于原始 speedrun.sh (8xH100, 3h) 的关键改动:
#   - 模型: d6 (depth=6) 替代 d24 — 适合在 16GB 显存内运行
#   - 精度: FP16 (T4 拥有 FP16 tensor cores，不支持 BF16/FP8)
#   - 无 DDP: 单卡运行，不使用 torchrun 多进程
#   - 窗口模式: 全上下文 ("L") — T4 上没有 FA3，滑动窗口非常缓慢
#   - 减少数据: 预训练使用 ~3 个 shards 替代 170 个
#   - 较小的 batch size: device_batch_size=4, total_batch_size=4096
#   - target-param-data-ratio=6 — 稍微欠拟合以节省时间
#   - 降低了评估频率以最大化训练吞吐量

# 1) 启动示例 (最简单):
# bash runs/speedrun_t4_zh.sh
# 2) 在 screen 会话中启动示例 (因为运行大约需要 2 小时):
# screen -L -Logfile runs/speedrun_t4_zh.log -S speedrun_t4 bash runs/speedrun_t4_zh.sh
# 3) 带 wandb 日志的启动示例:
# WANDB_RUN=speedrun_t4 screen -L -Logfile runs/speedrun_t4_zh.log -S speedrun_t4 bash runs/speedrun_t4_zh.sh

# 默认的中间 artifacts 目录在 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# 强制开启 FP16 精度 — T4 (SM 7.5) 拥有 FP16 tensor cores 但不支持 BF16
export NANOCHAT_DTYPE=float16
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 基于 uv 设置 Python venv 虚拟环境

# 安装 uv (如果尚未安装)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 创建 .venv 本地虚拟环境 (如果不存在)
[ -d ".venv" ] || uv venv
# 安装仓库依赖
uv sync --extra gpu
# 激活 venv，使 `python` 使用项目的 venv 而非系统自带 python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb 设置
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 重置报告
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器 (Tokenizer)

# 下载少量数据用于训练分词器 (~2B 字符 = 8 shards)
# 以及用于 d6 模型预训练的几个额外 shards
python -m nanochat.dataset -n 8
# 对于 ratio=6 的 d6，我们需要少得多的 tokens。在后台下载少量的额外
# shards。d6 有 ~7M 参数 => ~42M tokens => ~3 shards 就完全足够了。
# 为了保留安全余量，下载 10 shards。
python -m nanochat.dataset -n 10 &
DATASET_DOWNLOAD_PID=$!
# 在大约 2B 字符数据上，使用词表大小 2**15 = 32768 训练分词器
python -m scripts.tok_train
# 评估分词器
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基础模型 (预训练)
echo "正在等待数据集下载完成..."
wait $DATASET_DOWNLOAD_PID

# 在单张 T4 GPU 上运行 d6 模型
# 针对 T4 的关键改动:
#   --depth=6              : 小型模型以适应 16GB 显存并加快训练速度
#   --device-batch-size=4  : 受限于 T4 的 16GB 显存
#   --total-batch-size=4096: 由于是单卡，采用较小的 total batch (避免过多的梯度累加步数)
#   --target-param-data-ratio=6 : 稍微欠拟合以追求速度
#   --window-pattern L     : 全上下文注意力 (无 FA3, 使用 SDPA 的滑动窗口速度很慢)
#   --max-seq-len=1024     : 较短的上下文以减少显存和计算量
#   --eval-every=500       : 降低评估频率以节省时间
#   --core-metric-every=-1 : 训练期间跳过 CORE 评估 (训练后统一评估)
#   --sample-every=-1      : 训练期间跳过采样
#   无 --fp8               : T4 不支持 FP8
python -m scripts.base_train \
    --depth=6 \
    --target-param-data-ratio=6 \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --max-seq-len=1024 \
    --window-pattern=L \
    --eval-every=500 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run=$WANDB_RUN

# 评估模型: CORE 指标，train/val 上的 BPB (Bits Per Byte)，并生成样本
python -m scripts.base_eval --device-batch-size=4 --max-per-task=100

# -----------------------------------------------------------------------------
# SFT (教导模型对话特殊 tokens，工具使用，多项选择)

# 下载合成的身份对话数据
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 运行 SFT 并评估模型 (单卡，较小的 batch size，较少的数据轮数)
python -m scripts.chat_sft \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --max-seq-len=1024 \
    --mmlu-epochs=1 \
    --gsm8k-epochs=1 \
    --eval-every=500 \
    --chatcore-every=-1 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i sft --batch-size=4 --max-problems=100

# 在命令行通过 CLI 与模型对话！去掉 -p 可以进行交互式对话
# python -m scripts.chat_cli -p "天空为什么是蓝色的？"

# 或者更好的方式是，以类似 ChatGPT 的漂亮 WebUI 和您的模型对话
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 生成完整报告
python -m nanochat.report generate
