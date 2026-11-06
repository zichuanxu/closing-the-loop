import json, hashlib
import numpy as np, torch
from safetensors import safe_open

base = "/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
ck = "/work/ckpt/clc_sel/model.safetensors"
T = ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"]
out = {}
with safe_open(base, "pt") as fb, safe_open(ck, "pt") as fc:
    kb, kc = set(fb.keys()), set(fc.keys())
    out["keys_base"], out["keys_ckpt"] = len(kb), len(kc)
    out["only_base"], out["only_ckpt"] = sorted(kb - kc)[:10], sorted(kc - kb)[:10]
    diff = []
    for k in sorted(kb & kc):
        a, b = fb.get_tensor(k), fc.get_tensor(k)
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            diff.append(k)
    out["differing_tensors"] = diff
    out["tensor_detail"] = {}
    for k in T:
        a, b = fb.get_tensor(k), fc.get_tensor(k)
        d = (b.float() - a.float())
        colnz = (d != 0).any(0)
        rownz = (d != 0).any(1)
        s = torch.linalg.svdvals(d.double())
        out["tensor_detail"][k] = {"dtype": str(a.dtype), "shape": list(a.shape), "cols_changed": int(colnz.sum()),
                                   "rows_changed": int(rownz.sum()), "sv_top3": s[:3].tolist(),
                                   "rel_change": float(d.norm() / a.float().norm()),
                                   "sha256_bf16": hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        # compare against edits npz cast to bf16
        npz = np.load("/work/edits/clc_sel.npz")
        e = torch.from_numpy(npz[k]).to(torch.bfloat16)
        out["tensor_detail"][k]["equals_npz_bf16"] = bool(torch.equal(e, b))
    out["dtypes_all_bf16_same_as_base"] = True
ej = json.load(open("/work/ckpt/clc_sel/edits.json"))
for k in T:
    out["tensor_detail"][k]["matches_edits_json_sha"] = out["tensor_detail"][k]["sha256_bf16"] == ej["edited_bf16_sha256"][k]
mj = json.load(open("/work/edits/clc_sel.json"))
out["edit_args"] = {k: v for k, v in mj["args"].items()}
out["edit_cols_n"] = {k: len(v["cols"]) for k, v in mj["tensors"].items()}
print(json.dumps(out, indent=1))
json.dump(out, open("/work/audit_addendum4/audit_part2.json", "w"), indent=1)
