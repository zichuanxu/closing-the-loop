"""Per-source (gsm8k / math) accuracy, finished-answer accuracy, cut share and tokens for run dirs.
  python3 by_source.py runs/dev_ref_base runs/dev_loc_clc ...
"""
import collections, gzip, json, sys

for r in sys.argv[1:]:
    rs = [json.loads(l) for l in gzip.open(f"{r}/records.jsonl.gz", "rt")]
    by = collections.defaultdict(list)
    for x in rs:
        by[x["id"].split("_")[0]].append(x)
    by["all"] = rs
    out = []
    for s, v in sorted(by.items()):
        fin = [x for x in v if x["finish"] == "stop"]
        acc = sum(x["correct"] for x in v) / len(v)
        facc = sum(x["correct"] for x in fin) / max(1, len(fin))
        tok = sum(x["n_tokens"] for x in v) / len(v)
        out.append(f"{s}: n={len(v)} acc={acc:.3f} fin_acc={facc:.3f} cut={1 - len(fin) / len(v):.2f} tok={tok:.0f}")
    print(r.split("/")[-1], " | ".join(out))
