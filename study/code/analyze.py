"""Frozen analysis of loopdir_001 (protocol.md + addenda 1-3): per-arm stats, paired bootstrap differences, gates.

  python analyze.py --runs /work/runs --out /work/runs/analysis.json
"""
import argparse, gzip, json

import numpy as np

GROUPS = {  # benchmark -> arm -> run dir
    "math500": {"A": "m500_A_base", "B": "m500_B_astra", "C": "m500_C_crd", "D": "m500_D_random", "E": "m500_E_nothink",
                "F": "m500_F_astra_nothink", "G": "m500_G_crd_nothink", "H": "m500_H_crdloc", "I": "m500_I_clc",
                "J": "m500_J_clc_nothink", "I2": "m500_I2_clcsel", "P": "m500_P_rp", "A_bf": "m500_A_bf",
                "B_bf": "m500_B_bf", "I2_bf": "m500_I2_bf"},
    "gsm8k": {"K": "gsm_K_clc", "L": "gsm_L_astra", "K2": "gsm_K2_clcsel"},
}
PAIRS = {"math500": [("C", "A"), ("C", "B"), ("B", "A"), ("D", "A"), ("C", "E"), ("B", "E"), ("C", "D"), ("G", "E"),
                     ("F", "E"), ("H", "B"), ("H", "A"), ("H", "C"), ("I", "A"), ("I", "B"), ("I", "H"), ("I", "C"),
                     ("I", "E"), ("J", "E"), ("J", "F"), ("I2", "B"), ("I2", "I"), ("I2", "A"), ("I2", "E"),
                     ("I2", "P"), ("I2", "A_bf"), ("I2_bf", "B_bf"), ("I2_bf", "I2"), ("A_bf", "A"), ("B_bf", "B"),
                     ("I2_bf", "E"), ("P", "A")],
         "gsm8k": [("K", "L"), ("K2", "L"), ("K2", "K")]}


def load(path):
    return {r["id"]: r for r in map(json.loads, gzip.open(path, "rt"))}


def boot(x, n=10000, seed=0):
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def arm_stats(rs):
    fin = [r["correct"] for r in rs if r["finish"] == "stop"]
    return {"acc": float(np.mean([r["correct"] for r in rs])), "n": len(rs),
            "cut_share": float(np.mean([r["finish"] == "length" for r in rs])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rs])),
            "finished_no_box": int(sum(r["finish"] == "stop" and r["pred"] is None for r in rs)),
            "acc_on_finished": float(np.mean(fin)) if fin else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for bench, arms in GROUPS.items():
        data = {}
        for k, d in arms.items():
            try:
                data[k] = load(f"{a.runs}/{d}/records.jsonl.gz")
            except FileNotFoundError:
                pass
        if not data:
            continue
        g = {"arms": {}, "diffs": {}}
        for k, rows in data.items():
            g["arms"][k] = arm_stats(list(rows.values()))
        for x, y in PAIRS[bench]:
            if x in data and y in data:
                ids = sorted(set(data[x]) & set(data[y]))
                d = np.array([float(data[x][i]["correct"]) - float(data[y][i]["correct"]) for i in ids])
                m, lo, hi = boot(d)
                g["diffs"][f"{x}-{y}"] = {"n": len(ids), "mean_pts": 100 * m, "ci95_pts": [100 * lo, 100 * hi],
                                          "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
        res[bench] = g
    d = {**res.get("math500", {}).get("diffs", {}), **res.get("gsm8k", {}).get("diffs", {})}
    ok = lambda k, f: (k in d) and bool(f(d[k]))
    res["gates"] = {
        "H1_C-A>=20_lo>0": ok("C-A", lambda r: r["mean_pts"] >= 20 and r["ci95_pts"][0] > 0),
        "H2_C-B_lo>-5": ok("C-B", lambda r: r["ci95_pts"][0] > -5),
        "H3_D-A<5": ok("D-A", lambda r: r["mean_pts"] < 5),
        "H5_G-E>=3_lo>0": ok("G-E", lambda r: r["mean_pts"] >= 3 and r["ci95_pts"][0] > 0),
        "H7_I-A>=20_lo>0": ok("I-A", lambda r: r["mean_pts"] >= 20 and r["ci95_pts"][0] > 0),
        "H8_I-B_lo>-5": ok("I-B", lambda r: r["ci95_pts"][0] > -5),
        "H9_K-L_lo>-5": ok("K-L", lambda r: r["ci95_pts"][0] > -5),
        "H10_I-H_lo>0": ok("I-H", lambda r: r["ci95_pts"][0] > 0),
        "H11_K2-L_lo>-5": ok("K2-L", lambda r: r["ci95_pts"][0] > -5),
        "H12_K2-K>0_lo>0": ok("K2-K", lambda r: r["mean_pts"] > 0 and r["ci95_pts"][0] > 0),
        "H13_I2-B_lo>-5": ok("I2-B", lambda r: r["ci95_pts"][0] > -5),
        "sup_I2-B_lo>0": ok("I2-B", lambda r: r["ci95_pts"][0] > 0),
        "sup_K2-L_lo>0": ok("K2-L", lambda r: r["ci95_pts"][0] > 0),
        "H15_I2-P_lo>0": ok("I2-P", lambda r: r["ci95_pts"][0] > 0),
        "H16_I2-A_bf_lo>0": ok("I2-A_bf", lambda r: r["ci95_pts"][0] > 0),
        "H17_I2_bf-B_bf_lo>0": ok("I2_bf-B_bf", lambda r: r["ci95_pts"][0] > 0),
    }
    try:
        res["H4_cosines"] = json.load(open(f"{a.runs}/cosines.json"))
    except FileNotFoundError:
        pass
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
