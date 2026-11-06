#!/usr/bin/env bash
# chain 3: remaining MATH-500 arms. H after arm A frees its GPU; C, D (thinking) and G (non-thinking) after chain2 froze alpha.
cd /home/xuzichuan/weightspace/experiments/loopdir_001
w() { docker wait "xu_loopdir_$1" > "logs/$1.exit"; docker logs "xu_loopdir_$1" > "logs/$1.log" 2>&1; }
E="/work/code/gen_eval.py --data /work/data/math500.jsonl"
w m500_A_base
bash code/run_gpu.sh m500_H_crdloc 1 $E --ckpt /work/ckpt/crdloc_astra --out /work/runs/m500_H_crdloc --mem 0.4
until grep -q "chain2 done" logs/chain2.out; do sleep 60; done
AL=$(python3 -c "import json;print(json.load(open('runs/selected_alpha.json'))['selected_alpha'])")
w m500_B_astra
bash code/run_gpu.sh m500_C_crd 0 $E --ckpt /work/ckpt/crd_a$AL --out /work/runs/m500_C_crd --mem 0.4
sleep 180
bash code/run_gpu.sh m500_D_random 0 $E --ckpt /work/ckpt/random_a$AL --out /work/runs/m500_D_random --mem 0.4
w m500_H_crdloc
bash code/run_gpu.sh m500_G_crd_nothink 1 $E --ckpt /work/ckpt/crd_a$AL --out /work/runs/m500_G_crd_nothink --protocol nothink35 --max-tokens 8192 --mem 0.3
w m500_C_crd; w m500_D_random; w m500_G_crd_nothink
echo "chain3 done alpha=$AL"
