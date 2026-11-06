#!/usr/bin/env bash
# chain 8b = chain 8 (addendum 3 arms, unchanged arguments) with the retrying launcher; then the failed dev_full_brc_a1.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
bash code/launch_retry.sh m500_I_clc 0.4 --ckpt /work/ckpt/clcloc --data /work/data/math500.jsonl
bash code/launch_retry.sh m500_J_clc_nothink 0.3 --ckpt /work/ckpt/clcloc --data /work/data/math500.jsonl --protocol nothink35 --max-tokens 8192
bash code/launch_retry.sh gsm_K_clc 0.4 --ckpt /work/ckpt/clcloc --data /work/data/gsm8k_test.jsonl
bash code/launch_retry.sh gsm_L_astra 0.4 --ckpt /work/ckpt/astra_p50c --data /work/data/gsm8k_test.jsonl
bash code/launch_retry.sh dev_full_brc_a1 0.25 --ckpt /work/ckpt/full_brc_a1 --data /work/data/dev.jsonl --offset 64 --limit 128
for n in m500_I_clc m500_J_clc_nothink gsm_K_clc gsm_L_astra dev_full_brc_a1; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain8 done"
