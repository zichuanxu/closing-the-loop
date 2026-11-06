"""Automated repetition directions from the base model's own outputs (no hand-picked contexts, no gradients).

Rule (fixed before any edit is evaluated): in each base generation, find the FIRST verbatim block of L generated tokens
that recurs later without overlap (first occurrence at j, recurrence at i >= j + L). Teacher-force prompt + generation up
to i + L; at every layer's residual output take the mean state over the first-occurrence window and over the recurrence
window. Identical tokens, same problem, same text; what differs is whether the model is repeating itself.
u_l = unit(mean over texts of (recurrence - first)). Odd-indexed texts are held out to score how consistently the
direction separates each pair (paired sign accuracy and the mean projection gap in units of its pooled SD).

  python extract_dirs.py --records R.jsonl.gz [R2 ...] --data X.jsonl --ckpt DIR --out DIR [--L 64] [--max-prefix 16384]
"""
import argparse, gzip, json, os, sys

import numpy as np
import torch

sys.path.insert(0, "/authors/exp8_math/code")
import mathqa as Q  # noqa: E402


def first_repeat(ids, L):
    """(j, i): first L-token block of ids that recurs at i >= j + L, chosen by the smallest i; None if none."""
    seen = {}
    for i in range(len(ids) - L + 1):
        key = tuple(ids[i:i + L])
        j = seen.get(key)
        if j is not None and i >= j + L:
            return j, i
        seen.setdefault(key, i)
    return None


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--max-prefix", type=int, default=16384)
    ap.add_argument("--max-texts", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).cuda().eval()
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(a.data)}
    pairs, skipped = [], {"no_repeat": 0, "too_long": 0}
    for path in a.records:
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            prompt = tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(probs[r["id"]]["problem"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            g_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
            hit = first_repeat(g_ids, a.L)
            if hit is None:
                skipped["no_repeat"] += 1
                continue
            j, i = hit
            if len(p_ids) + i + a.L > a.max_prefix:
                skipped["too_long"] += 1
                continue
            pairs.append({"id": r["id"], "sample": r["sample"], "p": p_ids, "g": g_ids[:i + a.L], "j": j, "i": i,
                          "finish": r["finish"]})
    pairs = pairs[:a.max_texts]
    print(f"{len(pairs)} pairs, skipped {skipped}", flush=True)
    n_layers = model.config.num_hidden_layers
    first = np.zeros((len(pairs), n_layers, model.config.hidden_size), np.float32)
    rep = np.zeros_like(first)
    # down_proj inputs (MLP neuron activations), summed over pairs: [layer, intermediate] for first / recurrence windows
    act = {"first": np.zeros((n_layers, model.config.intermediate_size)), "rep": np.zeros((n_layers, model.config.intermediate_size))}
    cur = {}
    layers = model.model.layers
    hooks = [layers[l].mlp.down_proj.register_forward_hook(lambda m, inp, out, l=l: cur.__setitem__(l, inp[0][0].float()))
             for l in range(n_layers)]
    with torch.no_grad():
        for k, pr in enumerate(pairs):
            ids = torch.tensor([pr["p"] + pr["g"]], device="cuda")
            hs = model.model(ids, output_hidden_states=True).hidden_states  # hs[l+1] = residual output of layer l
            o = len(pr["p"])
            wf, wr = slice(o + pr["j"], o + pr["j"] + a.L), slice(o + pr["i"], o + pr["i"] + a.L)
            for l in range(n_layers):
                h = hs[l + 1][0].float()
                first[k, l] = h[wf].mean(0).cpu().numpy()
                rep[k, l] = h[wr].mean(0).cpu().numpy()
                act["first"][l] += cur[l][wf].mean(0).cpu().numpy()
                act["rep"][l] += cur[l][wr].mean(0).cpu().numpy()
            if k % 32 == 0:
                print(k, len(ids[0]), flush=True)
    for hk in hooks:
        hk.remove()
    diff = rep - first
    fit, held = np.arange(len(pairs)) % 2 == 0, np.arange(len(pairs)) % 2 == 1
    dirs = np.zeros((n_layers, diff.shape[2]), np.float32)
    stats = []
    for l in range(n_layers):
        u = unit(diff[fit, l].mean(0))
        dirs[l] = u
        proj = diff[held, l] @ u
        sd = np.sqrt(0.5 * ((first[held, l] @ u).var() + (rep[held, l] @ u).var()))
        stats.append({"layer": l, "heldout_sign_acc": float((proj > 0).mean()), "heldout_gap_sd": float(proj.mean() / sd),
                      "fit_norm_mean_diff": float(np.linalg.norm(diff[fit, l].mean(0))),
                      "resid_norm": float(np.linalg.norm(first[:, l], axis=1).mean())})
        print(json.dumps(stats[-1]), flush=True)
    full = np.stack([unit(diff[:, l].mean(0)) for l in range(n_layers)])  # all texts, used for editing
    np.savez(os.path.join(a.out, "directions.npz"), fit_half=dirs, all=full,
             act_first=(act["first"] / len(pairs)).astype(np.float32), act_rep=(act["rep"] / len(pairs)).astype(np.float32))
    np.savez_compressed(os.path.join(a.out, "window_means.npz"), first=first, rep=rep)
    json.dump({"pairs": [{k: v for k, v in p.items() if k not in ("p", "g")} | {"prefix_len": len(p["p"]) + len(p["g"])}
                         for p in pairs], "skipped": skipped, "L": a.L, "stats": stats,
               "records": a.records, "ckpt": a.ckpt}, open(os.path.join(a.out, "extract.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
