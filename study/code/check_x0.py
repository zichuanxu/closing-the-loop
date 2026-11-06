"""Diagnostic: does Astra's u + Astra's columns + alpha 6 in our formula rebuild Astra's released tensors?"""
import json, numpy as np, torch
from safetensors.torch import load_file
A = load_file("/work/ckpt/astra_p50c/model.safetensors"); B0 = load_file("/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
R = np.load("/work/edits/x0_astra_u_astra_cols.npz")
out = {}
for l in (15, 19):
    n = f"model.language_model.layers.{l}.mlp.down_proj.weight"
    a, b = A[n].float().numpy(), B0[n].float().numpy()
    r = torch.from_numpy(R[n]).to(torch.bfloat16).float().numpy()
    out[l] = {"bf16_exact_frac": float((a == r).mean()), "rel_err_delta": float(np.linalg.norm(a - r) / np.linalg.norm(a - b))}
print(json.dumps(out))
