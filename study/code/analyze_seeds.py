"""Addendum 5: per-problem mean correctness over seeds 0-2, paired bootstrap over problems (H14).
  python analyze_seeds.py --runs /work/runs --out /work/runs/analysis_seeds.json
"""
import argparse, gzip, json

import numpy as np

BENCH = {"math500": {"I2": ["m500_I2_clcsel", "m500_I2_clcsel_s1", "m500_I2_clcsel_s2"],
                     "B": ["m500_B_astra", "m500_B_astra_s1", "m500_B_astra_s2"]},
         # addendum 8: same keys (I2 = CLC-sel, B = Astra) so the analysis below is unchanged
         "gsm8k": {"I2": ["gsm_K2_clcsel", "gsm_K2_clcsel_s1", "gsm_K2_clcsel_s2"],
                   "B": ["gsm_L_astra", "gsm_L_astra_s1", "gsm_L_astra_s2"]}}


def load(path):
    return {r["id"]: r for r in map(json.loads, gzip.open(path, "rt"))}


def boot(x, n=10000, seed=0):
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return 100 * float(x.mean()), [100 * float(np.quantile(m, q)) for q in (0.025, 0.975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="math500", choices=tuple(BENCH))
    a = ap.parse_args()
    runs = {k: [load(f"{a.runs}/{d}/records.jsonl.gz") for d in v] for k, v in BENCH[a.bench].items()}
    ids = sorted(set.intersection(*[set(r) for v in runs.values() for r in v]))
    acc = {k: np.array([[float(r[i]["correct"]) for i in ids] for r in v]) for k, v in runs.items()}  # [seed, problem]
    res = {"n_problems": len(ids), "per_seed": {}, "pooled": {}}
    for s in range(3):
        cut = {k: float(np.mean([runs[k][s][i]["finish"] == "length" for i in ids])) for k in runs}
        m, ci = boot(acc["I2"][s] - acc["B"][s])
        res["per_seed"][s] = {"acc_I2": float(acc["I2"][s].mean()), "acc_B": float(acc["B"][s].mean()),
                              "cut_I2": cut["I2"], "cut_B": cut["B"], "I2-B": {"mean_pts": m, "ci95_pts": ci}}
    m, ci = boot(acc["I2"].mean(0) - acc["B"].mean(0))
    res["pooled"] = {"acc_I2": float(acc["I2"].mean()), "acc_B": float(acc["B"].mean()),
                     "I2-B": {"mean_pts": m, "ci95_pts": ci}}
    res["H14_pooled_I2-B_lo>0"] = bool(m > 0 and ci[0] > 0)   # math500: H14; gsm8k: superiority (reported)
    res["H20_pooled_lo>-5"] = bool(ci[0] > -5)                 # gsm8k: H20 non-inferiority
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
