"""Paired bootstrap, 10,000 resamples, seed 0 (same estimator as analyze_seeds.boot), for named run pairs B - A."""
import gzip, json, sys
import numpy as np
def load(d): return {r["id"]: float(r["correct"]) for r in map(json.loads, gzip.open(f"runs/{d}/records.jsonl.gz", "rt"))}
out = {}
for spec in sys.argv[1:]:
    a, b = spec.split(":")
    A, B = load(a), load(b); ids = sorted(set(A) & set(B))
    x = np.array([B[i] - A[i] for i in ids])
    m = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    out[f"{b}-{a}"] = [round(100 * x.mean(), 1), round(100 * np.quantile(m, .025), 1), round(100 * np.quantile(m, .975), 1)]
    print(f"{b} - {a}:", out[f"{b}-{a}"])
json.dump(out, open("runs/boot10k_paper.json", "w"), indent=1)
