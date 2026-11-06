#!/usr/bin/env bash
# chain 1: Astra MATH-500 arm after the base smoke frees GPU 0; then extraction + layer screen after extract generations.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
w() { docker wait "xu_loopdir_$1" > "logs/$1.exit"; docker logs "xu_loopdir_$1" > "logs/$1.log" 2>&1; }
w smoke_base
bash code/run_gpu.sh m500_B_astra 0 /work/code/gen_eval.py --ckpt /work/ckpt/astra_p50c --data /work/data/math500.jsonl --out /work/runs/m500_B_astra --mem 0.4
w gen_extract_base
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh extract 0 /work/code/extract_dirs.py --records /work/runs/gen_extract_base/records.jsonl.gz --data /work/data/extract.jsonl --ckpt /work/ckpt/base --out /work/runs/extract
w extract
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh screen 0 /work/code/screen_edits.py --extract /work/runs/extract --records /work/runs/gen_extract_base/records.jsonl.gz --data /work/data/extract.jsonl --ckpt /work/ckpt/base --out /work/runs/screen.json
w screen
echo chain1 done
