#!/usr/bin/env bash
# chain 2 (protocol steps 5-6): two screened layers, alpha in {2,4,6,8}, dev[64:192] generations; frozen selection.
set -euo pipefail
cd /home/xuzichuan/weightspace/experiments/loopdir_001
D=$PWD; A=/home/xuzichuan/weightspace/artifacts/external/lm_understand-logs; M=/home/xuzichuan/weightspace/artifacts/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17
LAYERS=$(python3 -c "import json;print(*json.load(open(\"runs/screen.json\"))[\"ranked\"][:2])")
echo "layers $LAYERS" | tee runs/selected_layers.txt
cpu() { docker run --rm --cpus=4 --memory=16g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NVIDIA_VISIBLE_DEVICES=void --network=none --user $(id -u):$(id -g) --mount type=bind,src=$D,dst=/work --mount type=bind,src=$A,dst=/authors,readonly --mount type=bind,src=$M,dst=/models/Qwen3.5-0.8B,readonly -w /work --entrypoint python3 xu_loopdir:20261104b "$@"; }
for AL in 2 4 6 8; do
  cpu code/make_edit.py --dirs runs/extract/directions.npz --layers $LAYERS --alpha $AL --k 64 --out edits/crd_a$AL.npz
  cpu code/build_ckpt.py ckpt/crd_a$AL --npz edits/crd_a$AL.npz
done
w() { docker wait "xu_loopdir_$1" > "logs/$1.exit"; docker logs "xu_loopdir_$1" > "logs/$1.log" 2>&1; }
gpu_for() { python3 - <<P
import subprocess
r=[l.split(",") for l in subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip().splitlines()]
print(min(r,key=lambda x:int(x[1]))[0].strip())
P
}
for AL in 2 4; do bash code/run_gpu.sh dev_crd_a$AL $(gpu_for) /work/code/gen_eval.py --ckpt /work/ckpt/crd_a$AL --data /work/data/dev.jsonl --offset 64 --limit 128 --out /work/runs/dev_crd_a$AL --mem 0.22; sleep 120; done
w dev_crd_a2; bash code/run_gpu.sh dev_crd_a6 $(gpu_for) /work/code/gen_eval.py --ckpt /work/ckpt/crd_a6 --data /work/data/dev.jsonl --offset 64 --limit 128 --out /work/runs/dev_crd_a6 --mem 0.22
w dev_crd_a4; bash code/run_gpu.sh dev_crd_a8 $(gpu_for) /work/code/gen_eval.py --ckpt /work/ckpt/crd_a8 --data /work/data/dev.jsonl --offset 64 --limit 128 --out /work/runs/dev_crd_a8 --mem 0.22
w dev_crd_a6; w dev_crd_a8
python3 - <<P | tee runs/selected_alpha.json
import json
r={a:json.load(open(f"runs/dev_crd_a{a}/summary.json")) for a in (2,4,6,8)}
best=max((2,4,6,8),key=lambda a:(r[a]["acc"],-a))
print(json.dumps({"by_alpha":{a:{k:r[a][k] for k in ("acc","cut_share","mean_tokens")} for a in r},"selected_alpha":best}))
P
AL=$(python3 -c "import json;print(json.load(open(\"runs/selected_alpha.json\"))[\"selected_alpha\"])")
sha256sum ckpt/crd_a$AL/model.safetensors edits/crd_a$AL.npz > runs/frozen_crd.sha256; cat ckpt/crd_a$AL/edits.json >> runs/frozen_crd.sha256
cpu code/make_edit.py --dirs runs/extract/directions.npz --layers $LAYERS --alpha $AL --k 64 --select weight --random-seed 0 --out edits/random_a$AL.npz
cpu code/build_ckpt.py ckpt/random_a$AL --npz edits/random_a$AL.npz
echo "chain2 done alpha=$AL layers=$LAYERS"
