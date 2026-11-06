#!/usr/bin/env bash
# chain 14 (addendum 8): GSM8K seeds 1, 2 for CLC-sel and Astra.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
G=/work/data/gsm8k_test.jsonl
for s in 1 2; do
  bash code/launch_retry.sh gsm_K2_clcsel_s$s 0.4 --ckpt /work/ckpt/clc_sel --data $G --seed $s
  bash code/launch_retry.sh gsm_L_astra_s$s 0.4 --ckpt /work/ckpt/astra_p50c --data $G --seed $s
done
echo "chain14 all launched"
for n in gsm_K2_clcsel_s1 gsm_L_astra_s1 gsm_K2_clcsel_s2 gsm_L_astra_s2; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain14 done"
