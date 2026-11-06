import gzip, json, sys, re, hashlib, os
import numpy as np

sys.path.insert(0, "/authors/exp8_math/code")
sys.path.insert(1, "/work/code")
import mathqa as Q

R = "/work/runs"
OUT = {}


def load(path):
    return [json.loads(l) for l in gzip.open(path, "rt")]


def jl(path):
    return [json.loads(l) for l in open(path)]


math = jl("/work/data/math500.jsonl")
gsm = jl("/work/data/gsm8k_test.jsonl")
res = jl("/work/data/reserve.jsonl")
ext = jl("/work/data/extract.jsonl")
gold = {r["id"]: r["answer"] for r in math + gsm + res}

# ---- 2. selection ----
sel = {}
for tag in ["R0_clc", "R1_astra", "R3_clc_a4", "R4_clc_a3", "R5_clc_sel", "R6_mix"]:
    recs = load(f"{R}/res_{tag}/records.jsonl.gz")
    ids = [r["id"] for r in recs]
    assert sorted(ids) == sorted(r["id"] for r in res), tag
    stored = np.mean([r["correct"] for r in recs])
    regr = []
    for r in recs:
        pred, _ = Q.extract(r["text"])
        regr.append(bool(Q.is_correct(pred, gold[r["id"]])))
    dis = sum(a != r["correct"] for a, r in zip(regr, recs))
    fin = [r["correct"] for r in recs if r["finish"] == "stop"]
    sel[tag] = {"n": len(recs), "acc_stored": float(stored), "acc_regraded": float(np.mean(regr)),
                "disagree": int(dis), "acc_finished": float(np.mean(fin)), "cut": float(np.mean([r["finish"] == "length" for r in recs]))}
    print("sel", tag, sel[tag], flush=True)
OUT["selection"] = sel

# ---- 3. grading ----
grad = {}
for name, data in [("m500_I2_clcsel", math), ("gsm_K2_clcsel", gsm)]:
    recs = load(f"{R}/{name}/records.jsonl.gz")
    g = {r["id"]: r["answer"] for r in data}
    ids = [r["id"] for r in recs]
    info = {"n": len(recs), "unique_ids": len(set(ids)), "ids_equal_dataset_order": ids == [r["id"] for r in data],
            "ids_equal_dataset_set": set(ids) == set(g)}
    dis = []
    pdis = 0
    for r in recs:
        pred, _ = Q.extract(r["text"])
        if pred != r["pred"]:
            pdis += 1
        c = bool(Q.is_correct(pred, g[r["id"]]))
        if c != r["correct"]:
            dis.append({"id": r["id"], "stored": r["correct"], "regraded": c, "pred": pred, "gold": g[r["id"]]})
    info["pred_disagree"] = pdis
    info["correct_disagree"] = len(dis)
    info["disagree_rows"] = dis[:20]
    info["acc_stored"] = float(np.mean([r["correct"] for r in recs]))
    info["acc_regraded"] = info["acc_stored"] + (sum((d["regraded"] - d["stored"]) for d in dis)) / len(recs)
    info["n_tokens_ok"] = None
    grad[name] = info
    print("grade", name, {k: v for k, v in info.items() if k != "disagree_rows"}, flush=True)
OUT["grading"] = grad

# ---- 4. stats ----
def L(name):
    return {r["id"]: r for r in load(f"{R}/{name}/records.jsonl.gz")}


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean() * 100), float(np.quantile(m, .025) * 100), float(np.quantile(m, .975) * 100)


arms = {"I2": "m500_I2_clcsel", "B": "m500_B_astra", "I": "m500_I_clc", "A": "m500_A_base", "E": "m500_E_nothink",
        "K2": "gsm_K2_clcsel", "L": "gsm_L_astra", "K": "gsm_K_clc"}
D = {k: L(v) for k, v in arms.items()}
# regrade comparison arms too (so differences use verified correctness)
regr_arm = {}
for k, v in D.items():
    g = gold
    dis = 0
    for i, r in v.items():
        pred, _ = Q.extract(r["text"])
        if bool(Q.is_correct(pred, g[i])) != r["correct"]:
            dis += 1
    regr_arm[k] = dis
    print("regrade arm", k, dis, flush=True)
OUT["regrade_comparison_arms_disagree"] = regr_arm
OUT["arm_stats"] = {}
for k, v in D.items():
    rs = list(v.values())
    fin = [r["correct"] for r in rs if r["finish"] == "stop"]
    OUT["arm_stats"][k] = {"n": len(rs), "acc": float(np.mean([r["correct"] for r in rs])),
                           "cut": float(np.mean([r["finish"] == "length" for r in rs])),
                           "mean_tokens": float(np.mean([r["n_tokens"] for r in rs])),
                           "acc_finished": float(np.mean(fin))}
orderM = [r["id"] for r in math]
orderG = [r["id"] for r in gsm]
stats = {}
for x, y, order in [("I2", "B", orderM), ("I2", "I", orderM), ("I2", "E", orderM), ("I2", "A", orderM),
                    ("K2", "L", orderG), ("K2", "K", orderG)]:
    assert set(D[x]) == set(D[y]) == set(order)
    ids_sorted = sorted(order)
    d1 = [float(D[x][i]["correct"]) - float(D[y][i]["correct"]) for i in ids_sorted]
    d2 = [float(D[x][i]["correct"]) - float(D[y][i]["correct"]) for i in order]
    m, lo, hi = boot(d1)
    m2, lo2, hi2 = boot(d2)
    stats[f"{x}-{y}"] = {"mean": m, "ci": [lo, hi], "ci_datasetorder": [lo2, hi2],
                         "wins": int(sum(v > 0 for v in d1)), "losses": int(sum(v < 0 for v in d1))}
    print("stat", x, y, stats[f"{x}-{y}"], flush=True)
OUT["stats"] = stats
st = json.load(open(f"{R}/analysis_a4.json"))
cmp = {}
dd = {**st["math500"]["diffs"], **st["gsm8k"]["diffs"]}
for k, v in stats.items():
    s = dd[k]
    cmp[k] = {"max_abs_diff": max(abs(v["mean"] - s["mean_pts"]), abs(v["ci"][0] - s["ci95_pts"][0]), abs(v["ci"][1] - s["ci95_pts"][1])),
              "wins_losses_match": (v["wins"], v["losses"]) == (s["wins"], s["losses"])}
OUT["compare_analysis_a4"] = cmp
print(cmp)
gates = {"H11_K2-L_lo>-5": stats["K2-L"]["ci"][0] > -5,
         "H12_K2-K>0_lo>0": stats["K2-K"]["mean"] > 0 and stats["K2-K"]["ci"][0] > 0,
         "H13_I2-B_lo>-5": stats["I2-B"]["ci"][0] > -5,
         "sup_I2-B_lo>0": stats["I2-B"]["ci"][0] > 0, "sup_K2-L_lo>0": stats["K2-L"]["ci"][0] > 0}
OUT["gates"] = gates
OUT["gates_match_analysis_a4"] = {k: st["gates"][k] == v for k, v in gates.items()}
print(gates, OUT["gates_match_analysis_a4"])

# ---- 5. leakage ----
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def sh(s, k=6):
    w = norm(s).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


pool = {"reserve": res, "extract": ext}
test = {"math500": math, "gsm8k_test": gsm}
leak = {"counts": {k: len(v) for k, v in {**pool, **test}.items()}}
for pn, prow in pool.items():
    for tn, trow in test.items():
        idov = len({r["id"] for r in prow} & {r["id"] for r in trow})
        tx = {norm(r["problem"]) for r in trow}
        txov = sum(norm(r["problem"]) in tx for r in prow)
        # near-duplicate: 6-gram shingle containment
        inv = {}
        tsh = [sh(r["problem"]) for r in trow]
        for j, s in enumerate(tsh):
            for g_ in s:
                inv.setdefault(g_, set()).add(j)
        near = []
        for r in prow:
            s = sh(r["problem"])
            cnt = {}
            for g_ in s:
                for j in inv.get(g_, ()):
                    cnt[j] = cnt.get(j, 0) + 1
            if cnt:
                j, c = max(cnt.items(), key=lambda t: t[1])
                frac = c / max(1, min(len(s), len(tsh[j])))
                if frac >= 0.7:
                    near.append({"pool_id": r["id"], "test_id": trow[j]["id"], "frac": round(frac, 2)})
        leak[f"{pn}_vs_{tn}"] = {"id_overlap": idov, "exact_norm_text_overlap": txov, "near_dup_ge0.7": len(near), "near_examples": near[:5]}
        print("leak", pn, tn, leak[f"{pn}_vs_{tn}"], flush=True)
leak["reserve_vs_extract_id_overlap"] = len({r["id"] for r in res} & {r["id"] for r in ext})
leak["dup_ids_within"] = {k: len(v) - len({r["id"] for r in v}) for k, v in {**pool, **test}.items()}
leak["reserve_sources"] = {"gsm8k_train": sum(r["id"].startswith("gsm8k_train") for r in res), "math_train": sum(r["id"].startswith("math_train") for r in res)}
leak["extract_sources"] = {k: sum(r["id"].startswith(k) for r in ext) for k in ["gsm8k_train", "math_train", "gsm8k_test", "test/"]}
OUT["leakage"] = leak

json.dump(OUT, open("/work/audit_addendum4/audit_part1.json", "w"), indent=1, default=str)
