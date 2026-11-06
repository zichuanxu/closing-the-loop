"""Addendum 9 analysis: paired bootstrap (10,000 resamples, seed 0) for B - A pairs; missing runs skipped; per-run stats."""
import gzip, json, os
import numpy as np
PAIRS = ["m500_N_nowait:m500_I2_clcsel", "gsm_N_nowait:gsm_K2_clcsel", "m500_A_base:m500_N_nowait",
         "m500_I2_clcsel:m500_N2_clcsel_nowait", "m500_B_astra:m500_I4_auto2", "gsm_L_astra:gsm_K4_auto2",
         "m500_I2_clcsel:m500_I4_auto2", "gsm_K2_clcsel:gsm_K4_auto2", "m500_I3_auto:m500_I4_auto2"]
def load(d): return {r["id"]: r for r in map(json.loads, gzip.open(f"runs/{d}/records.jsonl.gz", "rt"))}
out, seen = {"pairs": {}, "runs": {}}, set()
for spec in PAIRS:
    a, b = spec.split(":")
    if not all(os.path.exists(f"runs/{d}/records.jsonl.gz") for d in (a, b)):
        continue
    A, B = load(a), load(b); ids = sorted(set(A) & set(B))
    x = np.array([float(B[i]["correct"]) - float(A[i]["correct"]) for i in ids])
    m = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    out["pairs"][f"{b}-{a}"] = [round(100 * x.mean(), 1), round(100 * np.quantile(m, .025), 1), round(100 * np.quantile(m, .975), 1)]
    print(f"{b} - {a}:", out["pairs"][f"{b}-{a}"])
    for d, R in ((a, A), (b, B)):
        if d in seen: continue
        seen.add(d); rs = list(R.values()); fin = [r for r in rs if r["finish"] == "stop"]
        out["runs"][d] = {"n": len(rs), "acc": round(100 * np.mean([r["correct"] for r in rs]), 1),
                          "trunc": round(100 * (1 - len(fin) / len(rs)), 1),
                          "fin_acc": round(100 * np.mean([r["correct"] for r in fin]), 1), "tokens": round(np.mean([r["n_tokens"] for r in rs]))}
        print(d, out["runs"][d])
json.dump(out, open("runs/boot_add9.json", "w"), indent=1)
