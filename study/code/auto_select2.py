"""Addendum 9 pick v2 (plain python3): best screen accuracy over stage B (alpha*), C (pair) and D (alpha*/2).
Ties -> finished-answer accuracy, then smaller total alpha, then lower layer.  python3 auto_select2.py OUT.json"""
import gzip, json, sys

alpha = {int(k): v["alpha_star"] for k, v in json.load(open("runs/auto_alpha.json")).items()}
pair = json.load(open("runs/auto_pick.json"))["pair"]


def stats(run):
    rs = [json.loads(l) for l in gzip.open(f"runs/{run}/records.jsonl.gz", "rt")]
    fin = [r for r in rs if r["finish"] == "stop"]
    return sum(r["correct"] for r in rs) / len(rs), sum(r["correct"] for r in fin) / max(1, len(fin))


cands = [(f"scr_L{l}", [l], [alpha[l]], f"/work/ckpt/auto_L{l}") for l in range(1, 24, 2)]
cands.append(("scr_pair", pair["layers"], pair["alphas"], "/work/ckpt/auto_pair"))
cands += [(f"scr_h{l}", [l], [alpha[l] / 2], f"/work/ckpt/auto_h{l}") for l in (9, 11, 13, 15, 17, 19)]
rows = []
for run, layers, alphas, ck in cands:
    acc, fin = stats(run)
    rows.append({"run": run, "layers": layers, "alphas": alphas, "ckpt": ck, "acc": acc, "fin_acc": fin})
best = sorted(rows, key=lambda r: (-r["acc"], -r["fin_acc"], sum(r["alphas"]), r["layers"][0]))[0]
json.dump({"rows": rows, "pick": best, "same_as_addendum7": best["run"] == "scr_L13"}, open(sys.argv[1], "w"), indent=1)
for r in sorted(rows, key=lambda r: -r["acc"])[:8]:
    print(r["run"], round(100 * r["acc"], 1), round(100 * r["fin_acc"], 1), [round(a, 2) for a in r["alphas"]])
print("pick", best["run"], best["ckpt"])
