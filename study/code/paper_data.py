"""Collect per-run summary numbers for the paper's tables and figures (plain python3).
  python3 code/paper_data.py OUT.json        # every runs/*/records.jsonl.gz
"""
import glob, gzip, json, sys

out = {}
for p in sorted(glob.glob("runs/*/records.jsonl.gz")):
    run = p.split("/")[1]
    rs = [json.loads(l) for l in gzip.open(p, "rt")]
    fin = [r for r in rs if r["finish"] == "stop"]
    out[run] = {"n": len(rs), "acc": sum(r["correct"] for r in rs) / len(rs),
                "cut": 1 - len(fin) / len(rs),
                "fin_acc": sum(r["correct"] for r in fin) / max(1, len(fin)),
                "mean_tok": sum(r["n_tokens"] for r in rs) / len(rs),
                "forced": sum(bool(r.get("forced")) for r in rs) / len(rs)}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
