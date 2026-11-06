"""BRC stage 2: matched-prefix branch-contrast directions. For each loop-onset prefix with both kinds of sampled
continuation, teacher-force prefix + each continuation (one batch: equal lengths) and average the residual state over the
continuation tokens at every layer. d_p = mean(repeat branches) - mean(diverge branches), same prefix and position;
u_l = unit(mean_p d_p). Same output format as extract_dirs.py (act_rep = neuron activations on repeat branches).

  python branch_dirs.py --branches F.jsonl.gz --ckpt DIR --out DIR
"""
import argparse, gzip, json, os

import numpy as np
import torch


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batch", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).cuda().eval()
    nl, H, I = model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size
    cur = {}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(
        lambda m, inp, out, l=l: cur.__setitem__(l, inp[0].float())) for l in range(nl)]
    rows = [json.loads(x) for x in gzip.open(a.branches, "rt")]
    used = [r for r in rows if {c["label"] for c in r["conts"]} == {"repeat", "diverge"}]
    print(f"{len(used)} of {len(rows)} prefixes have both kinds", flush=True)
    D = np.zeros((len(used), nl, H), np.float32)
    act = {"repeat": np.zeros((nl, I)), "diverge": np.zeros((nl, I))}
    cnt = {"repeat": 0, "diverge": 0}
    with torch.no_grad():
        for k, r in enumerate(used):
            conts = r["conts"]
            L = len(conts[0]["ids"])
            o = len(r["prefix"])
            lab = np.array([c["label"] for c in conts])
            M = np.zeros((len(conts), nl, H), np.float32)
            A = np.zeros((len(conts), nl, I), np.float32)
            for s in range(0, len(conts), a.max_batch):  # ponytail: chunks of max_batch (OOM at 8 x 16k under co-tenancy)
                x = torch.tensor([r["prefix"] + c["ids"] for c in conts[s:s + a.max_batch]], device="cuda")
                hs = model.model(x, output_hidden_states=True).hidden_states
                for l in range(nl):
                    M[s:s + len(x), l] = hs[l + 1][:, o:o + L].float().mean(1).cpu().numpy()
                    A[s:s + len(x), l] = cur[l][:, o:o + L].mean(1).cpu().numpy()
                del hs
            for l in range(nl):
                D[k, l] = M[lab == "repeat", l].mean(0) - M[lab == "diverge", l].mean(0)
                for kind in act:
                    act[kind][l] += A[lab == kind, l].sum(0)
            for kind in cnt:
                cnt[kind] += int((lab == kind).sum())
            if k % 32 == 0:
                print(k, o, flush=True)
    for hk in hooks:
        hk.remove()
    fit, held = np.arange(len(used)) % 2 == 0, np.arange(len(used)) % 2 == 1
    stats, fit_dirs = [], np.zeros((nl, H), np.float32)
    for l in range(nl):
        u = unit(D[fit, l].mean(0))
        fit_dirs[l] = u
        proj = D[held, l] @ u
        stats.append({"layer": l, "heldout_sign_acc": float((proj > 0).mean()),
                      "heldout_t": float(proj.mean() / (proj.std() / np.sqrt(len(proj))))})
        print(json.dumps(stats[-1]), flush=True)
    np.savez(os.path.join(a.out, "directions.npz"), all=np.stack([unit(D[:, l].mean(0)) for l in range(nl)]),
             fit_half=fit_dirs, act_rep=(act["repeat"] / cnt["repeat"]).astype(np.float32),
             act_first=(act["diverge"] / cnt["diverge"]).astype(np.float32))
    json.dump({"prefixes_used": len(used), "prefixes_total": len(rows), "branch_counts": cnt, "stats": stats,
               "branches": a.branches}, open(os.path.join(a.out, "extract.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
