"""Addendum 7: pre-closure statistics for the closed-form strength. Same windows as closure_dirs.py (the L tokens before
the last '</think>' of finished extract answers): mean residual state per layer and mean down_proj input per layer.

  python closure_stats.py --records R.jsonl.gz --data X.jsonl --ckpt DIR --out DIR [--L 64]
"""
import argparse, gzip, json, os, sys

import numpy as np
import torch

sys.path.insert(0, "/authors/exp8_math/code")
import mathqa as Q  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
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
    cur = {}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(
        lambda m, inp, out, l=l: cur.__setitem__(l, inp[0][0].float())) for l in range(nl)]
    resid, act, n = [], np.zeros((nl, model.config.intermediate_size)), 0
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
            resid.append(np.stack([hs[l + 1][0, p - a.L:p].float().mean(0).cpu().numpy() for l in range(nl)]))
            for l in range(nl):
                act[l] += cur[l][p - a.L:p].mean(0).cpu().numpy()
            n += 1
    for h in hooks:
        h.remove()
    np.savez(os.path.join(a.out, "closure_stats.npz"), pre_resid=np.stack(resid).mean(0).astype(np.float32),
             act_pre=(act / n).astype(np.float32))
    json.dump({"n_finished_used": n}, open(os.path.join(a.out, "closure_stats.json"), "w"))
    print({"n_finished_used": n}, flush=True)


if __name__ == "__main__":
    main()
