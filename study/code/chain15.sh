#!/usr/bin/env bash
# chain 15 (addendum 7 confirmation): wait for the automated pick, then run it on MATH-500 (I3) and GSM8K test (K3).
cd /home/xuzichuan/weightspace/experiments/loopdir_001
until grep -q "chain13 screen done" logs/chain13.out; do sleep 120; done
read ck same < <(python3 -c "
import json; p = json.load(open('runs/auto_pick.json'))['pick']
same = p['layers'] == [15, 19] and all(abs(a - 6) <= 0.5 for a in p['alphas'])
print('/work/ckpt/auto_pair' if len(p['layers']) == 2 else '/work/ckpt/auto_L%d' % p['layers'][0], int(same))")
echo "$(date -u +%FT%TZ) pick ckpt $ck same_as_clcsel=$same"
[ "$same" = 1 ] && { echo "pick equals CLC-sel; I2/K2 are its confirmation (addendum 7)"; echo "chain15 done"; exit 0; }
bash code/launch_retry.sh m500_I3_auto 0.4 --ckpt $ck --data /work/data/math500.jsonl
bash code/launch_retry.sh gsm_K3_auto 0.4 --ckpt $ck --data /work/data/gsm8k_test.jsonl
for n in m500_I3_auto gsm_K3_auto; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain15 done"
