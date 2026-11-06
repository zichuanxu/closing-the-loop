"""Overlap with Astra's 64 columns per layer for automated column rules at layers 15/19.
branch_contrast = (act on repeating branches - act on diverging branches of the same prefix) * u.W_c  (BRC acts)."""
import json, numpy as np
from safetensors.torch import load_file
C = np.load("/work/runs/extract/directions.npz"); B = np.load("/work/runs/brc/directions.npz")
W0 = load_file("/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"); WA = load_file("/work/ckpt/astra_p50c/model.safetensors")
out = {}
for l in (15, 19):
    n = f"model.language_model.layers.{l}.mlp.down_proj.weight"
    w0 = W0[n].float().numpy().astype(np.float64); d = WA[n].float().numpy() - w0
    astra = set(np.flatnonzero(np.abs(d).sum(0) > 0))
    row = {}
    for uk, U in (("crd_u", C["all"][l]), ("brc_u", B["all"][l])):
        s = U @ w0
        for rk, v in {"act_rep_crd": C["act_rep"][l] * s, "branch_contrast": (B["act_rep"][l] - B["act_first"][l]) * s,
                      "abs_branch_contrast": np.abs((B["act_rep"][l] - B["act_first"][l]) * s)}.items():
            row[f"{uk}/{rk}"] = len(astra & set(np.argsort(-v)[:64].tolist()))
    out[l] = row
print(json.dumps(out, indent=1))
