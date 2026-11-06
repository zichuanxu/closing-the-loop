#!/usr/bin/env bash
# chain 13 (addendum 7): closed-form alpha per layer, single-layer generation screen, pair, pick.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
D=$PWD; A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs; M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
cpu() { docker run --rm --cpus=4 --memory=16g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NVIDIA_VISIBLE_DEVICES=void --network=none --user $(id -u):$(id -g) --mount type=bind,src=$D,dst=/work --mount type=bind,src=$A,dst=/authors,readonly --mount type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly -w /work --entrypoint python3 xu_loopdir:20261104b "$@" 2>&1 | tail -3; }
waitmem() { while true; do for g in 0 1; do [ "$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)" -gt $1 ] && { echo $g; return; }; done; sleep 60; done; }
E="--dirs runs/clc/directions.npz --u-from dirs --k 64 --act-first runs/extract/directions.npz --alpha 0"
SCR="--data /work/data/screen.jsonl"
LAYERS="1 3 5 7 9 11 13 15 17 19 21 23"
# 1. pre-closure stats (GPU, HF) and closed-form alpha (CPU)
g=$(waitmem 14000)
IMAGE=xu_loopdir:20261104b bash code/run_gpu.sh clc_stats $g /work/code/closure_stats.py --records /work/runs/gen_extract_base/records.jsonl.gz --data /work/data/extract.jsonl --ckpt /work/ckpt/base --out /work/runs/clc_stats
docker wait xu_loopdir_clc_stats > logs/clc_stats.exit; docker logs xu_loopdir_clc_stats > logs/clc_stats.log 2>&1
cpu code/auto_alpha.py --clc runs/clc/directions.npz --extract runs/extract --stats runs/clc_stats/closure_stats.npz --out runs/auto_alpha.json
[ -s runs/auto_alpha.json ] || { echo "auto_alpha failed"; exit 1; }
python3 -c "import json;print('alpha*', {k: round(v['alpha_star'], 2) for k, v in json.load(open('runs/auto_alpha.json')).items()})"
# 2. single-layer edits at alpha*, screen runs (+ base)
for l in $LAYERS; do
  al=$(python3 -c "import json;print(json.load(open('runs/auto_alpha.json'))['$l']['alpha_star'])")
  cpu code/make_edit.py $E --layers $l --alphas $al --out edits/auto_L$l.npz
  cpu code/build_ckpt.py ckpt/auto_L$l --npz edits/auto_L$l.npz
done
echo "$(date -u +%FT%TZ) stage B edits built"
bash code/launch_retry.sh scr_base 0.25 --ckpt /work/ckpt/base $SCR
for l in $LAYERS; do bash code/launch_retry.sh scr_L$l 0.25 --ckpt /work/ckpt/auto_L$l $SCR; done
for n in scr_base $(for l in $LAYERS; do echo scr_L$l; done); do docker wait xu_loopdir_$n > logs/$n.exit; docker logs xu_loopdir_$n > logs/$n.log 2>&1; done
# 3. pair of the two best, screen, pick
python3 code/auto_select.py top2 runs/auto_top2.json
read l1 l2 a1 a2 < <(python3 -c "import json;p=json.load(open('runs/auto_top2.json'))['pair'];print(*p['layers'],*p['alphas'])")
cpu code/make_edit.py $E --layers $l1 $l2 --alphas $a1 $a2 --out edits/auto_pair.npz
cpu code/build_ckpt.py ckpt/auto_pair --npz edits/auto_pair.npz
bash code/launch_retry.sh scr_pair 0.25 --ckpt /work/ckpt/auto_pair $SCR
docker wait xu_loopdir_scr_pair > logs/scr_pair.exit; docker logs xu_loopdir_scr_pair > logs/scr_pair.log 2>&1
python3 code/auto_select.py pick runs/auto_pick.json | tee logs/auto_pick.txt
echo "chain13 screen done"
