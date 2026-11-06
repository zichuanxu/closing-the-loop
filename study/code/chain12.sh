#!/usr/bin/env bash
# chain 12 (addendum 6): repetition-penalty selection on reserve, budget forcing on MATH-500, then the penalty pick on MATH-500.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
R=/work/data/reserve.jsonl; M=/work/data/math500.jsonl
bash code/launch_retry.sh res_rp105 0.4 --ckpt /work/ckpt/base --data $R --rep-penalty 1.05
bash code/launch_retry.sh res_rp110 0.4 --ckpt /work/ckpt/base --data $R --rep-penalty 1.1
bash code/launch_retry.sh res_rp120 0.4 --ckpt /work/ckpt/base --data $R --rep-penalty 1.2
bash code/launch_retry.sh m500_A_bf 0.4 --ckpt /work/ckpt/base --data $M --force-close 30720
bash code/launch_retry.sh m500_I2_bf 0.4 --ckpt /work/ckpt/clc_sel --data $M --force-close 30720
bash code/launch_retry.sh m500_B_bf 0.4 --ckpt /work/ckpt/astra_p50c --data $M --force-close 30720
for n in res_rp105 res_rp110 res_rp120; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
pick=$(python3 -c "
import json
r={p:json.load(open(f\"runs/res_rp{t}/summary.json\"))[\"acc\"] for p,t in ((1.05,\"105\"),(1.1,\"110\"),(1.2,\"120\"))}
best=max(r.values()); print(min(p for p in r if r[p]==best))")
echo "$(date -u +%FT%TZ) addendum 6 repetition-penalty pick on reserve: $pick ($(for t in 105 110 120; do python3 -c "import json;print(\"rp$t\",json.load(open(\"runs/res_rp$t/summary.json\"))[\"acc\"])"; done | tr "\n" " "))" | tee runs/selected_rp.txt
bash code/launch_retry.sh m500_P_rp 0.4 --ckpt /work/ckpt/base --data $M --rep-penalty $pick
for n in m500_A_bf m500_I2_bf m500_B_bf m500_P_rp; do docker wait "xu_loopdir_$n" > "logs/$n.exit"; docker logs "xu_loopdir_$n" > "logs/$n.log" 2>&1; done
echo "chain12 done"
