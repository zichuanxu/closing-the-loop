"""Logit lens of edit directions: tokens most promoted (+) / suppressed (-) by E (g * v), g = final RMSNorm gain."""
import json, numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
sd = load_file("/work/ckpt/base/model.safetensors"); tok = AutoTokenizer.from_pretrained("/work/ckpt/base")
E = sd["model.language_model.embed_tokens.weight"].float(); g = sd["model.language_model.norm.weight"].float()
C = np.load("/work/runs/extract/directions.npz")["all"]
ast = json.load(open("/work/edits/astra_reconstructed.json"))["tensors"]
def top(v, k=12):
    lg = E @ (g * torch.tensor(v, dtype=torch.float32)) if g.mean() > 0.5 else E @ ((1 + g) * torch.tensor(v, dtype=torch.float32))
    return {"+": [tok.decode([i]) for i in lg.topk(k).indices.tolist()], "-": [tok.decode([i]) for i in (-lg).topk(k).indices.tolist()]}
out = {"norm_gain_mean": float(g.mean())}
for l in (15, 19):
    ua = np.array(ast[f"model.language_model.layers.{l}.mlp.down_proj.weight"]["u"]); c = C[l] / np.linalg.norm(C[l])
    if ua @ c < 0: ua = -ua
    r = ua - (ua @ c) * c; r /= np.linalg.norm(r)
    out[f"L{l}"] = {"astra": top(ua), "crd": top(c), "astra_orth_to_crd": top(r)}
print(json.dumps(out, indent=1, ensure_ascii=False)); json.dump(out, open("/work/runs/lens.json", "w"), indent=1, ensure_ascii=False)
