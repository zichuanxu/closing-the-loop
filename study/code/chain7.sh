#!/usr/bin/env bash
# chain 7 (exploratory, dev[64:192]): column-free edits W' = (I - alpha u u^T) W at layers 15,19 (k = 0, all columns),
# automated directions only. Serialized, memory-gated launches as in chain6; waits for chain6's launches to finish.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
D=$PWD; A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs; M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
cpu() { docker run --rm --cpus=4 --memory=16g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NVIDIA_VISIBLE_DEVICES=void --network=none --user $(id -u):$(id -g) --mount type=bind,src=$D,dst=/work --mount type=bind,src=$A,dst=/authors,readonly --mount type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly -w /work --entrypoint python3 xu_loopdir:20261104b "$@" 2>&1 | tail -1; }
for spec in "brc 1" "brc 2" "crd 2"; do set -- $spec
  src=$([ $1 = crd ] && echo extract || echo brc)
  cpu code/make_edit.py --dirs runs/$src/directions.npz --layers 15 19 --alpha $2 --k 0 --out edits/full_${1}_a$2.npz
  cpu code/build_ckpt.py ckpt/full_${1}_a$2 --npz edits/full_${1}_a$2.npz
done
until grep -q "launched dev_loc_clc" logs/chain6.out 2>/dev/null || docker ps -a --format "{{.Names}}" | grep -q xu_loopdir_dev_loc_clc; do sleep 60; done
sleep 180
launch() {
  local need=$((10240 + 3072))
  while true; do for g in 0 1; do
    if [ "$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)" -gt "$need" ]; then
      bash code/run_gpu.sh "$1" $g /work/code/gen_eval.py --ckpt "/work/ckpt/$2" --data /work/data/dev.jsonl --offset 64 --limit 128 --out "/work/runs/$1" --mem 0.25
      sleep 180; return; fi; done; sleep 60; done; }
launch dev_full_brc_a1 full_brc_a1
launch dev_full_brc_a2 full_brc_a2
launch dev_full_crd_a2 full_crd_a2
for n in dev_full_brc_a1 dev_full_brc_a2 dev_full_crd_a2; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain7 done"
