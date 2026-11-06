#!/usr/bin/env bash
# chain 8 (addendum 3 confirmation): I, J on MATH-500; K, L on GSM8K test. Serialized memory-gated launches.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
launch() {  # name gpu-mem-fraction args...
  local name=$1 mem=$2; shift 2
  local need=$(python3 -c "print(int($mem*40960)+3072)")
  while true; do for g in 0 1; do
    if [ "$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)" -gt "$need" ]; then
      bash code/run_gpu.sh "$name" $g /work/code/gen_eval.py --mem $mem --out "/work/runs/$name" "$@"
      sleep 180; return; fi; done; sleep 60; done; }
launch m500_I_clc 0.4 --ckpt /work/ckpt/clcloc --data /work/data/math500.jsonl
launch m500_J_clc_nothink 0.3 --ckpt /work/ckpt/clcloc --data /work/data/math500.jsonl --protocol nothink35 --max-tokens 8192
launch gsm_K_clc 0.4 --ckpt /work/ckpt/clcloc --data /work/data/gsm8k_test.jsonl
launch gsm_L_astra 0.4 --ckpt /work/ckpt/astra_p50c --data /work/data/gsm8k_test.jsonl
for n in m500_I_clc m500_J_clc_nothink gsm_K_clc gsm_L_astra; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain8 done"
