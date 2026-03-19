#!/bin/bash

# 本脚本是为单张 A100 (40GB) 显卡量身定制的大模型训练流程脚本（预训练 + 微调）。
# 原版 speedrun.sh 是为 8xH100 设计的，这里的参数已经大幅缩减了模型层数和并行设定，
# 以确保能在预算有限（如 6 美元）和单卡单节点的环境下，在 2-4 小时内顺利跑通全流程。

# 1) 最简单的执行方式:
# bash runs/speedrun_a100_zh.sh
# 2) 推荐在 screen 会话中运行（防止网络断开导致训练中止）:
# screen -L -Logfile runs/speedrun_a100.log -S speedrun bash runs/speedrun_a100_zh.sh

# 设置环境变量，中间产物默认保存在 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python 虚拟环境与依赖配置 (使用 uv)

# 检查是否安装了 uv，没有则自动安装
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 如果当前目录没有 .venv，则创建一个新的虚拟环境
[ -d ".venv" ] || uv venv
# 下载并同步带 gpu 支持的依赖包
uv sync --extra gpu
# 激活虚拟环境，接下来的 python 命令都会使用这个环境
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb 可视化监控设置
# 如果想在网页端直观看到训练曲线，建议使用 wandb。
# 使用前请确保你已经在终端里执行过 `wandb login`
if [ -z "$WANDB_RUN" ]; then
    # 默认使用 "dummy"，会自动跳过向 wandb 发送日志
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 清空之前的生成报告，并在 base dir 中重新初始化一个报告的头部信息
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器 (Tokenizer) 准备阶段

# 下载前 8 个预训练数据分块（大约 20 亿个字符，约 800MB）
python -m nanochat.dataset -n 8
# 在后台启动下载更多数据。原脚本需要 170 个数据块，这里因为是训缩小版的跑通 Demo，
# 只需要少量的验证数据，下载 20 个足够满足 depth=12 的使用量了，能省下很多硬盘和网络开销。
python -m nanochat.dataset -n 20 &
DATASET_DOWNLOAD_PID=$!

# 使用前面下载的 8 个文件里的 20 亿个字符，训练你自己的 32768 词表的分词器
python -m scripts.tok_train
# 评估并打印分词器的压缩率等表现
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基础模型（预训练，也就是从头教它语言规律的阶段）
echo "等待后台的预训练数据分块下载完成..."
wait $DATASET_DOWNLOAD_PID

echo "开始基础大模型预训练..."
# 【核心修改区】
# - 改成了单卡并行 (--nproc_per_node=1)
# - 将深度缩小至 d12 (--depth=12)，极大加快单卡训练速度
# - 批次设为 16 (--device-batch-size=16)，刚好填满或适合单张 A100 的 40G 显存
# - 删除了不支持/没必要的 --fp8 加速配置
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=16 \
    --run=$WANDB_RUN

# 评估刚训好的基础模型能力 (CORE 分数、Loss损失 等)
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT 有监督微调（教会模型聊天格式、不要乱说废话，并注入人格）

echo "下载合成的对话数据..."
# 下载用于 SFT 的身份认知和其他小众合成对话数据
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

echo "开始 SFT 监督微调..."
# 启动聊天微调（同样修改为了单卡和适配的 batch size）
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN

# 评估聊天的 SFT 模型
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# =============================================================================
# 全部训练大功告成！接下来的步骤你可以手动在终端运行：

# 1. 命令行聊天测试（不用图形界面，直接在终端里回复）：
# python -m scripts.chat_cli -p "天空为什么是蓝色的？"

# 2. ChatGPT 风格的 Web 网页版聊天（必须在本地激活虚拟环境后执行该命令，再打开提示的网址）：
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 收集所有截断的日志，生成属于你的完整大模型训练报告
python -m nanochat.report generate
echo "全流程完成！属于你的袖珍大模型已出炉！"
