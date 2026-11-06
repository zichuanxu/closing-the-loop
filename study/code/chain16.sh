#!/usr/bin/env bash
# chain 16 (addendum 9): half-strength screen (stage D), pick v2, NoWait runs, confirmation of pick v2 if new.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
D=$PWD; A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs; M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
cpu() { docker run --rm --cpus=4 --memory=16g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NVIDIA_VISIBLE_DEVICES=void --network=none --user $(id -u):$(id -g) --mount type=bind,src=$D,dst=/work --mount type=bind,src=$A,dst=/authors,readonly --mount type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly -w /work --entrypoint python3 xu_loopdir:20261104b "$@" 2>&1 | tail -3; }
E="--dirs runs/clc/directions.npz --u-from dirs --k 64 --act-first runs/extract/directions.npz --alpha 0"
SCR="--data /work/data/screen.jsonl"
H="9 11 13 15 17 19"
fin() { for n in "$@"; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done; }
# stage D edits (alpha*/2) and screen
for l in $H; do
  al=$(python3 -c "import json;print(json.load(open('runs/auto_alpha.json'))['$l']['alpha_star'] / 2)")
  cpu code/make_edit.py $E --layers $l --alphas $al --out edits/auto_h$l.npz
  cpu code/build_ckpt.py ckpt/auto_h$l --npz edits/auto_h$l.npz
done
echo "$(date -u +%FT%TZ) stage D edits built"
for l in $H; do bash code/launch_retry.sh scr_h$l 0.25 --ckpt /work/ckpt/auto_h$l $SCR; done
# NoWait on MATH-500 (H21) can start while the screen runs
bash code/launch_retry.sh m500_N_nowait 0.4 --ckpt /work/ckpt/base --data /work/data/math500.jsonl --nowait
fin $(for l in $H; do echo scr_h$l; done)
python3 code/auto_select2.py runs/auto_pick2.json | tee logs/auto_pick2.txt
read ck same < <(python3 -c "import json;p=json.load(open('runs/auto_pick2.json'));print(p['pick']['ckpt'], int(p['same_as_addendum7']))")
echo "$(date -u +%FT%TZ) pick2 $ck same_as_addendum7=$same"
if [ "$same" = 0 ]; then
  bash code/launch_retry.sh m500_I4_auto2 0.4 --ckpt $ck --data /work/data/math500.jsonl
  bash code/launch_retry.sh gsm_K4_auto2 0.4 --ckpt $ck --data /work/data/gsm8k_test.jsonl
fi
bash code/launch_retry.sh gsm_N_nowait 0.4 --ckpt /work/ckpt/base --data /work/data/gsm8k_test.jsonl --nowait
bash code/launch_retry.sh m500_N2_clcsel_nowait 0.4 --ckpt /work/ckpt/clc_sel --data /work/data/math500.jsonl --nowait
fin m500_N_nowait gsm_N_nowait m500_N2_clcsel_nowait
[ "$same" = 0 ] && fin m500_I4_auto2 gsm_K4_auto2
echo "chain16 done"
