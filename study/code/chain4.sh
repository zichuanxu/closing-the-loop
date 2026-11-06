#!/usr/bin/env bash
# chain 4 (exploratory diagnostics, dev[64:192] only): base / Astra references and the direction-vs-column swaps.
# Launches each run on a GPU only when its free memory exceeds the request plus a 3 GiB margin.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
launch() {  # name ckpt mem
  local need=$(python3 -c "print(int($3*40960)+3072)")
  while true; do
    for g in 0 1; do
      free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$free" -gt "$need" ]; then
        bash code/run_gpu.sh "$1" $g /work/code/gen_eval.py --ckpt "/work/ckpt/$2" --data /work/data/dev.jsonl --offset 64 --limit 128 --out "/work/runs/$1" --mem $3
        sleep 150; return
      fi
    done
    sleep 60
  done
}
launch dev_ref_base base 0.25
launch dev_ref_astra astra_p50c 0.25
launch dev_x1_astra_u_crd_cols x1 0.25
launch dev_x2_crd_u_astra_cols x2 0.25
for n in dev_ref_base dev_ref_astra dev_x1_astra_u_crd_cols dev_x2_crd_u_astra_cols; do
  docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1
done
echo "chain4 done"
