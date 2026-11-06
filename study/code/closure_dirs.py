"""Closure-contrast directions (CLC): u_l = unit(mean recurrence-window state - mean pre-closure state), where pre-closure =
the L tokens just before the base model's own '</think>' in extract answers that finished. With the Astra edit form
(alpha > 1 inverts the u component), loop states are pushed toward the state in which the model concludes.
Recurrence states and act_rep are reused from extract_dirs (window_means.npz / directions.npz). No labels of correctness.

  python closure_dirs.py --extract DIR --records R.jsonl.gz --data X.jsonl --ckpt DIR --out DIR [--L 64]
"""
import argparse, gzip, json, os, sys

import numpy as np
import torch

sys.path.insert(0, "/authors/exp8_math/code")
import mathqa as Q  # noqa: E402


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).cuda().eval()
    nl = model.config.num_hidden_layers
    close_id = tok.convert_tokens_to_ids("</think>")
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(a.data)}
    fin = [r for r in map(json.loads, gzip.open(a.records, "rt")) if r["finish"] == "stop" and r["closed_think"]]
    pre = []
    with torch.no_grad():
        for r in fin:
            prompt = tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(probs[r["id"]]["problem"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
            ids = tok(prompt, add_special_tokens=False)["input_ids"] + tok(r["text"], add_special_tokens=False)["input_ids"]
            pos = [k for k, t in enumerate(ids) if t == close_id]
            if not pos or pos[-1] < a.L:
                continue
            p = pos[-1]
            hs = model.model(torch.tensor([ids[:p]], device="cuda"), output_hidden_states=True).hidden_states
            pre.append(np.stack([hs[l + 1][0, p - a.L:p].float().mean(0).cpu().numpy() for l in range(nl)]))
    pre = np.stack(pre)                                         # [n_fin, layers, H]
    rep = np.load(f"{a.extract}/window_means.npz")["rep"]       # [pairs, layers, H]
    act_rep = np.load(f"{a.extract}/directions.npz")["act_rep"]
    dirs = np.stack([unit(rep[:, l].mean(0) - pre[:, l].mean(0)) for l in range(nl)])
    fit = np.stack([unit(rep[0::2, l].mean(0) - pre[0::2, l].mean(0)) for l in range(nl)])
    stats = []
    for l in range(nl):  # held-out separation: odd halves projected on the even-half direction
        pr, pp = rep[1::2, l] @ fit[l], pre[1::2, l] @ fit[l]
        thr = 0.5 * (pr.mean() + pp.mean())
        stats.append({"layer": l, "heldout_acc": float(0.5 * ((pr > thr).mean() + (pp < thr).mean()))})
    np.savez(os.path.join(a.out, "directions.npz"), all=dirs, fit_half=fit, act_rep=act_rep)
    json.dump({"n_finished_used": len(pre), "n_recurrence": len(rep), "stats": stats},
              open(os.path.join(a.out, "extract.json"), "w"), indent=1)
    print(json.dumps({"n_finished_used": len(pre), "n_recurrence": len(rep)}), [round(s["heldout_acc"], 3) for s in stats], flush=True)


if __name__ == "__main__":
    main()
