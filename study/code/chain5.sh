#!/usr/bin/env bash
# chain 5 (exploratory, BRC development): branch sampling then branch-contrast directions, on GPUs with free memory.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
pick() {  # MiB needed -> echo a GPU index once one has that much free
  while true; do for g in 0 1; do
    [ "$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)" -gt "$1" ] && { echo $g; return; }
  done; sleep 60; done; }
w() { docker wait "xu_loopdir_$1" > "logs/$1.exit"; docker logs "xu_loopdir_$1" > "logs/$1.log" 2>&1; }
G=$(pick 15360)
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh brc_sample $G /work/code/branch_sample.py --extract /work/runs/extract --records /work/runs/gen_extract_base/records.jsonl.gz --data /work/data/extract.jsonl --ckpt /work/ckpt/base --out /work/runs/brc_branches.jsonl.gz --mem 0.3
w brc_sample
G=$(pick 16384)
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh brc_dirs $G /work/code/branch_dirs.py --branches /work/runs/brc_branches.jsonl.gz --ckpt /work/ckpt/base --out /work/runs/brc
w brc_dirs
echo "chain5 done"
