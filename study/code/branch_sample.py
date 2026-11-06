"""BRC stage 1: at each loop onset found by extract_dirs (prefix = prompt + generation[:i], where the block first seen at j
recurs), sample K continuations of L tokens from the base model (authors' thinking sampling, seeded). A continuation is
'repeat' if at least half of its tokens match the earlier block position-wise, else 'diverge'.

  python branch_sample.py --extract DIR --records R.jsonl.gz --data X.jsonl --ckpt DIR --out F.jsonl.gz [--K 8]
"""
import argparse, gzip, json, sys

sys.path.insert(0, "/authors/exp8_math/code")
import mathqa as Q  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mem", type=float, default=0.3)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    ex = json.load(open(f"{a.extract}/extract.json"))
    L = ex["L"]
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(a.data)}
    recs = {(r["id"], r["sample"]): r for r in map(json.loads, gzip.open(a.records, "rt"))}
    items = []
    for k, p in enumerate(ex["pairs"]):
        prompt = tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(probs[p["id"]]["problem"])}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        g = tok(recs[(p["id"], p["sample"])]["text"], add_special_tokens=False)["input_ids"]
        items.append({"pair": k, "prefix": pi + g[:p["i"]], "block": g[p["j"]:p["j"] + L]})
    llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=a.mem, max_model_len=20480, seed=0,
              enable_prefix_caching=True)
    sp = SamplingParams(n=a.K, max_tokens=L, min_tokens=L, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                        seed=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=it["prefix"]) for it in items], sp)
    n_rep = n_div = 0
    with gzip.open(a.out, "wt") as f:
        for it, o in zip(items, outs):
            conts = []
            for s in o.outputs:
                c = list(s.token_ids)[:L]
                match = sum(x == y for x, y in zip(c, it["block"])) / L
                lab = "repeat" if match >= 0.5 else "diverge"
                n_rep += lab == "repeat"; n_div += lab == "diverge"
                conts.append({"ids": c, "match": match, "label": lab})
            f.write(json.dumps({"pair": it["pair"], "prefix": it["prefix"], "conts": conts}) + "\n")
    print(json.dumps({"prefixes": len(items), "repeat": n_rep, "diverge": n_div}), flush=True)


if __name__ == "__main__":
    main()
