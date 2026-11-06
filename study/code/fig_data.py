"""Figure inputs (plain python3): paired outcome decomposition of B - A in accuracy points.
  python3 code/fig_data.py OUT.json
Categories per problem: both finished / A truncated, B finished / B truncated, A finished / both truncated.
"""
import gzip, json, os, sys

PAIRS = {f"math500_s{s}": (f"m500_B_astra{t}", f"m500_I2_clcsel{t}") for s, t in ((0, ""), (1, "_s1"), (2, "_s2"))}
PAIRS.update({f"gsm8k_s{s}": (f"gsm_L_astra{t}", f"gsm_K2_clcsel{t}") for s, t in ((0, ""), (1, "_s1"), (2, "_s2"))})
PAIRS["math500_bf"] = ("m500_B_bf", "m500_I2_bf")


def load(d):
    return {r["id"]: r for r in map(json.loads, gzip.open(f"runs/{d}/records.jsonl.gz", "rt"))}


out = {}
for name, (a, b) in PAIRS.items():
    if not all(os.path.exists(f"runs/{d}/records.jsonl.gz") for d in (a, b)):
        continue
    A, B = load(a), load(b)
    ids = sorted(set(A) & set(B))
    fin = lambda r: r["finish"] == "stop"
    cat = {"both_finished": 0.0, "rescued": 0.0, "lost": 0.0, "both_truncated": 0.0}
    cnt = dict.fromkeys(cat, 0)
    for i in ids:
        k = ("both_finished" if fin(A[i]) and fin(B[i]) else "rescued" if fin(B[i]) else
             "lost" if fin(A[i]) else "both_truncated")
        cat[k] += (B[i]["correct"] - A[i]["correct"]) * 100 / len(ids)
        cnt[k] += 1
    out[name] = {"points": cat, "counts": cnt, "net": sum(cat.values()), "n": len(ids)}
    assert abs(out[name]["net"] - 100 * (sum(B[i]["correct"] for i in ids) - sum(A[i]["correct"] for i in ids)) / len(ids)) < 1e-9
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out.items():
    print(k, {c: round(p, 2) for c, p in v["points"].items()}, "net", round(v["net"], 2), v["counts"])
