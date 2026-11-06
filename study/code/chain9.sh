#!/usr/bin/env bash
# chain 9 (addendum 4): build closure-fix candidates, freeze, run all on reserve (256) with the retrying launcher.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
D=$PWD; A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs; M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
cpu() { docker run --rm --cpus=4 --memory=16g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NVIDIA_VISIBLE_DEVICES=void --network=none --user $(id -u):$(id -g) --mount type=bind,src=$D,dst=/work --mount type=bind,src=$A,dst=/authors,readonly --mount type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly -w /work --entrypoint python3 xu_loopdir:20261104b "$@" 2>&1 | tail -1; }
C="--dirs runs/clc/directions.npz --u-from dirs --layers 15 19 --k 64"
cpu code/make_edit.py $C --alpha 4 --out edits/clc_a4.npz
cpu code/make_edit.py $C --alpha 3 --out edits/clc_a3.npz
cpu code/make_edit.py $C --alpha 6 --act-first runs/extract/directions.npz --out edits/clc_sel.npz
cpu code/make_edit.py $C --alpha 6 --mix runs/extract/directions.npz --out edits/mix.npz
cpu code/make_edit.py $C --alpha 6 --out edits/clc_recheck.npz   # must reproduce edits/clcloc.npz
for e in clc_a4 clc_a3 clc_sel mix; do cpu code/build_ckpt.py ckpt/$e --npz edits/$e.npz; done
sha256sum addendum_4_closure_fix.md code/make_edit.py edits/{clc_a4,clc_a3,clc_sel,mix,clcloc,clc_recheck}.npz ckpt/{clc_a4,clc_a3,clc_sel,mix}/model.safetensors > runs/frozen_fix.sha256
date -u +%FT%TZ >> runs/frozen_fix.sha256
for spec in "res_R1_astra astra_p50c" "res_R0_clc clcloc" "res_R3_clc_a4 clc_a4" "res_R4_clc_a3 clc_a3" "res_R5_clc_sel clc_sel" "res_R6_mix mix"; do set -- $spec
  bash code/launch_retry.sh $1 0.4 --ckpt /work/ckpt/$2 --data /work/data/reserve.jsonl; done
for n in res_R1_astra res_R0_clc res_R3_clc_a4 res_R4_clc_a3 res_R5_clc_sel res_R6_mix; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain9 done"
