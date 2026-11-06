#!/usr/bin/env bash
# chain 6 (exploratory, dev[64:192]): build BRC/CLC edits at Astra's location (15,19; alpha 6; k 64; activation columns),
# relaunch the two failed references, then the location-matched direction comparisons. One launch at a time, gated on
# free memory with a 3 GiB margin and a 180 s gap so vLLM profiling never overlaps another of our engine starts.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
D=$PWD; A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs; M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
cpu() { docker run --rm --cpus=4 --memory=16g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NVIDIA_VISIBLE_DEVICES=void --network=none --user $(id -u):$(id -g) --mount type=bind,src=$D,dst=/work --mount type=bind,src=$A,dst=/authors,readonly --mount type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly -w /work --entrypoint python3 xu_loopdir:20261104b "$@" 2>&1 | tail -1; }
for k in brc clc; do
  cpu code/make_edit.py --dirs runs/$k/directions.npz --layers 15 19 --alpha 6 --k 64 --out edits/${k}loc.npz
  cpu code/build_ckpt.py ckpt/${k}loc --npz edits/${k}loc.npz
done
launch() {  # name ckpt
  local need=$((10240 + 3072))
  while true; do for g in 0 1; do
    if [ "$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)" -gt "$need" ]; then
      bash code/run_gpu.sh "$1" $g /work/code/gen_eval.py --ckpt "/work/ckpt/$2" --data /work/data/dev.jsonl --offset 64 --limit 128 --out "/work/runs/$1" --mem 0.25
      sleep 180; return; fi; done; sleep 60; done; }
for n in dev_ref_astra dev_x1_astra_u_crd_cols; do
  docker logs xu_loopdir_$n > logs/$n.attempt1_failed_mem.log 2>&1; docker container rm $(docker inspect -f {{.Id}} xu_loopdir_$n) >/dev/null; mv runs/$n runs/$n.attempt1_failed
done
launch dev_ref_astra astra_p50c
launch dev_x1_astra_u_crd_cols x1
launch dev_loc_crd crdloc_astra
launch dev_loc_brc brcloc
launch dev_loc_clc clcloc
for n in dev_ref_base dev_ref_astra dev_x1_astra_u_crd_cols dev_x2_crd_u_astra_cols dev_loc_crd dev_loc_brc dev_loc_clc; do
  docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1
done
echo "chain6 done"
