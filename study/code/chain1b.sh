#!/usr/bin/env bash
# chain 1b: the same screen as chain1 (protocol step 4), layers split over both GPUs, the identical ranking rule; then chain2.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
w() { docker wait "xu_loopdir_$1" > "logs/$1.exit"; docker logs "xu_loopdir_$1" > "logs/$1.log" 2>&1; }
w extract
S="/work/code/screen_edits.py --extract /work/runs/extract --records /work/runs/gen_extract_base/records.jsonl.gz --data /work/data/extract.jsonl --ckpt /work/ckpt/base"
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh screen_lo 0 $S --out /work/runs/screen_lo.json --layers 0 1 2 3 4 5 6 7 8 9 10 11
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh screen_hi 1 $S --out /work/runs/screen_hi.json --layers 12 13 14 15 16 17 18 19 20 21 22 23
w screen_lo; w screen_hi
python3 code/merge_screen.py
echo "chain1b screen merged"
bash code/chain2.sh > logs/chain2.out 2>&1
echo "chain1b done"
