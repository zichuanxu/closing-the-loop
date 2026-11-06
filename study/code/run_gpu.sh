#!/usr/bin/env bash
# Launch one detached GPU job of this study on the GPU server.  usage: run_gpu.sh NAME GPU SCRIPT [args...]
# Mounts: study dir rw at /work, authors' logs ro, the model ro. Caches inside the study (Triton needs exec tmp).
set -euo pipefail
NAME=$1; GPU=$2; shift 2
D=/home/xuzichuan/weightspace/experiments/loopdir_001
A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs
M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
mkdir -p "$D/cache/tmp" "$D/cache/home" "$D/logs"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$D/logs/$NAME.gpu_pre.txt"
docker run -d --name "xu_loopdir_$NAME" --label owner=xuzichuan --label purpose=loopdir_001 \
  --cpus=8 --memory=48g --memory-swap=48g --shm-size=8g --pids-limit=4096 --gpus "device=$GPU" \
  --user "$(id -u):$(id -g)" -e HOME=/work/cache/home -e TMPDIR=/work/cache/tmp \
  -e TRITON_CACHE_DIR=/work/cache/triton -e VLLM_CACHE_ROOT=/work/cache/vllm -e XDG_CACHE_HOME=/work/cache/xdg \
  -e OMP_NUM_THREADS=4 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  --mount "type=bind,src=$D,dst=/work" --mount "type=bind,src=$A,dst=/authors,readonly" \
  --mount "type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly" \
  -w /work --entrypoint python3 ${IMAGE:-xu_loopdir:20261104} -u "$@"
echo "$(date -u +%FT%TZ) $NAME gpu=$GPU $*" >> "$D/logs/launches.txt"
