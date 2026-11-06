"""Teacher-forced screen of single-layer edits (cheap, no generation): for each layer l, apply the Astra-form edit
(alpha, k, activation-ranked columns) to that layer's down_proj and measure the mean change in log-probability
(nats/token) of
  rep  - the recurrence windows of the HELD-OUT half of extraction pairs (odd index; the direction used the even half),
  prod - 64-token windows inside the thinking of base outputs that FINISHED (no repetition labels involved).
A good loop edit lowers rep and leaves prod. Selection rule (fixed in the protocol): layers with prod >= -0.01, ranked
by rep (most negative first).

  python screen_edits.py --extract DIR --records R.jsonl.gz --data X.jsonl --ckpt DIR --out F.json [--alpha 6 --k 64]
"""
import argparse, gzip, json, random, sys

import numpy as np
import torch

sys.path.insert(0, "/authors/exp8_math/code")
import mathqa as Q  # noqa: E402

sys.path.insert(0, "/work/code")
from make_edit import edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=6)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-texts", type=int, default=48)
    ap.add_argument("--layers", type=int, nargs="*")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).cuda().eval()
    ex = json.load(open(f"{a.extract}/extract.json"))
    Z = np.load(f"{a.extract}/directions.npz")
    L = ex["L"]
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(a.data)}
    recs = {(r["id"], r["sample"]): r for r in map(json.loads, gzip.open(a.records, "rt"))}

    def prompt_ids(pid):
        p = tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(probs[pid]["problem"])}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        return tok(p, add_special_tokens=False)["input_ids"]

    items = []  # (kind, ids, window start)
    for p in ex["pairs"][1::2][:a.n_texts]:
        g = tok(recs[(p["id"], p["sample"])]["text"], add_special_tokens=False)["input_ids"][:p["i"] + L]
        pi = prompt_ids(p["id"])
        items.append(("rep", pi + g, len(pi) + p["i"]))
    rnd = random.Random(0)
    fin = [r for r in recs.values() if r["finish"] == "stop" and r["closed_think"]]
    rnd.shuffle(fin)
    for r in fin[:a.n_texts]:
        g = tok(r["text"].split("</think>")[0], add_special_tokens=False)["input_ids"]
        if len(g) < 2 * L:
            continue
        s = rnd.randrange(L, len(g) - L)
        pi = prompt_ids(r["id"])
        items.append(("prod", pi + g[:s + L], len(pi) + s))
    print({k: sum(i[0] == k for i in items) for k in ("rep", "prod")}, flush=True)

    @torch.no_grad()
    def score():
        out = {"rep": [], "prod": []}
        for kind, ids, s in items:
            x = torch.tensor([ids], device="cuda")
            h = model.model(x).last_hidden_state[0, s - 1:s + L - 1]
            lp = torch.log_softmax(model.lm_head(h).float(), -1)
            out[kind].append(lp.gather(1, x[0, s:s + L, None]).mean().item())
        return {k: np.array(v) for k, v in out.items()}

    base = score()
    res = {"base_mean": {k: float(v.mean()) for k, v in base.items()}, "alpha": a.alpha, "k": a.k, "layers": []}
    layers = a.layers if a.layers else range(model.config.num_hidden_layers)
    for l in layers:
        W = model.model.layers[l].mlp.down_proj.weight
        W0 = W.data.clone()
        u = Z["all"][l].astype(np.float64)
        We, cols = edit(W0.float().cpu().numpy().astype(np.float64), u, a.alpha, a.k, Z["act_rep"][l].astype(np.float64))
        W.data.copy_(torch.from_numpy(We).to(W.dtype))
        s = score()
        W.data.copy_(W0)
        d = {k: s[k] - base[k] for k in s}
        row = {"layer": l, "d_rep": float(d["rep"].mean()), "d_rep_se": float(d["rep"].std() / np.sqrt(len(d["rep"]))),
               "d_prod": float(d["prod"].mean()), "d_prod_se": float(d["prod"].std() / np.sqrt(len(d["prod"])))}
        res["layers"].append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in res["layers"] if r["d_prod"] >= -0.01]
    res["ranked"] = [r["layer"] for r in sorted(ok, key=lambda r: r["d_rep"])]
    json.dump(res, open(a.out, "w"), indent=1)
    print("ranked", res["ranked"])


if __name__ == "__main__":
    main()
