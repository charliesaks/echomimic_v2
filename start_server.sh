#!/bin/bash
# Start the EchoMimicV2 server with memory-safe settings.
# Use CPU to avoid MPS OOM crashes on Apple Silicon.
# CPU inference is slower (~5-20 min/clip) but stable.
#
# MPS settings (faster but risks system crash — only use with 32GB+ RAM):
#   ECHOMIMIC_DEVICE=mps ECHOMIMIC_SIZE=512 ./start_server.sh

cd "$(dirname "$0")"

export ECHOMIMIC_DEVICE="${ECHOMIMIC_DEVICE:-cpu}"
export ECHOMIMIC_SIZE="${ECHOMIMIC_SIZE:-256}"
export ECHOMIMIC_MAX_GEN_FRAMES="${ECHOMIMIC_MAX_GEN_FRAMES:-16}"
export ECHOMIMIC_CONTEXT_FRAMES="${ECHOMIMIC_CONTEXT_FRAMES:-4}"
export ECHOMIMIC_STEPS="${ECHOMIMIC_STEPS:-15}"
# Prevent MPS from caching too much memory even if device is overridden to mps
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3

exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001
