#!/usr/bin/env bash
# chain 11 (addendum 5): seeds 1, 2 for Astra and CLC-sel on MATH-500.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
for s in 1 2; do
  bash code/launch_retry.sh m500_I2_clcsel_s$s 0.4 --ckpt /work/ckpt/clc_sel --data /work/data/math500.jsonl --seed $s
  bash code/launch_retry.sh m500_B_astra_s$s 0.4 --ckpt /work/ckpt/astra_p50c --data /work/data/math500.jsonl --seed $s
done
for n in m500_I2_clcsel_s1 m500_B_astra_s1 m500_I2_clcsel_s2 m500_B_astra_s2; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain11 done"
