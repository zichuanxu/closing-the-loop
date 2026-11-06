"""|cos| between direction sets (CRD, BRC, CLC) and Astra's recovered u at layers 15/19; least-squares decomposition of
Astra's u on span{CRD, CLC} (how much of it the automated directions explain)."""
import json, numpy as np
S = {k: np.load(f"/work/runs/{d}/directions.npz")["all"] for k, d in (("crd", "extract"), ("brc", "brc"), ("clc", "clc"))}
ast = json.load(open("/work/edits/astra_reconstructed.json"))["tensors"]
out = {}
for l in (15, 19):
    ua = np.array(ast[f"model.language_model.layers.{l}.mlp.down_proj.weight"]["u"])
    row = {f"{k}_astra": abs(float(v[l] @ ua)) for k, v in S.items()}
    row["crd_clc"] = abs(float(S["crd"][l] @ S["clc"][l]))
    X = np.stack([S["crd"][l], S["clc"][l]], 1)
    coef, *_ = np.linalg.lstsq(X, ua, rcond=None)
    row["span_crd_clc_R2"] = float(1 - np.sum((ua - X @ coef) ** 2) / np.sum(ua ** 2))
    out[f"L{l}"] = row
print(json.dumps(out, indent=1)); json.dump(out, open("/work/runs/cosines.json", "w"), indent=1)
