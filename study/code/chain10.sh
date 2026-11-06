#!/usr/bin/env bash
# chain 10 (addendum 4 confirmation): pick clc_sel on MATH-500 (I2) and GSM8K test (K2).
cd /home/xuzichuan/weightspace/experiments/loopdir_001
bash code/launch_retry.sh m500_I2_clcsel 0.4 --ckpt /work/ckpt/clc_sel --data /work/data/math500.jsonl
bash code/launch_retry.sh gsm_K2_clcsel 0.4 --ckpt /work/ckpt/clc_sel --data /work/data/gsm8k_test.jsonl
for n in m500_I2_clcsel gsm_K2_clcsel; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain10 done"
