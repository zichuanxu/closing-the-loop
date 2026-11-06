"""Addendum 7 selection (plain python3, no numpy).
  python3 auto_select.py top2 OUT.json      -> best two single layers from runs/scr_L*/ and their pair alphas
  python3 auto_select.py pick OUT.json      -> pair vs best single, writes the pick
"""
import gzip, json, sys

LAYERS = list(range(1, 24, 2))


def stats(run):
    rs = [json.loads(l) for l in gzip.open(f"runs/{run}/records.jsonl.gz", "rt")]
    fin = [r for r in rs if r["finish"] == "stop"]
    return {"acc": sum(r["correct"] for r in rs) / len(rs),
            "fin_acc": sum(r["correct"] for r in fin) / max(1, len(fin)), "n": len(rs)}


mode, out = sys.argv[1], sys.argv[2]
alpha = {int(k): v["alpha_star"] for k, v in json.load(open("runs/auto_alpha.json")).items()}
single = {l: stats(f"scr_L{l}") for l in LAYERS}
order = sorted(LAYERS, key=lambda l: (-single[l]["acc"], -single[l]["fin_acc"], l))
res = {"single": single, "order": order, "base": stats("scr_base"), "alpha_star": alpha}
res["failed"] = all(single[l]["acc"] < res["base"]["acc"] for l in LAYERS)
l1, l2 = sorted(order[:2])
res["pair"] = {"layers": [l1, l2], "alphas": [alpha[l1] / 2, alpha[l2] / 2]}
if mode == "pick":
    res["pair_stats"] = stats("scr_pair")
    best = order[0]
    if res["pair_stats"]["acc"] > single[best]["acc"]:
        res["pick"] = res["pair"]
    else:
        res["pick"] = {"layers": [best], "alphas": [alpha[best]]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "single"}))
for l in order:
    print(l, round(alpha[l], 2), single[l])
