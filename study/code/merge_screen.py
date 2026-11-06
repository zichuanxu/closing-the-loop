"""Merge the two half screens (layers 0-11, 12-23) with screen_edits.py's own ranking rule."""
import json

a, b = json.load(open("runs/screen_lo.json")), json.load(open("runs/screen_hi.json"))
rows = a["layers"] + b["layers"]
ok = [r for r in rows if r["d_prod"] >= -0.01]
json.dump({"base_mean": [a["base_mean"], b["base_mean"]], "alpha": a["alpha"], "k": a["k"], "layers": rows,
           "ranked": [r["layer"] for r in sorted(ok, key=lambda r: r["d_rep"])],
           "merged_from": ["screen_lo.json", "screen_hi.json"]}, open("runs/screen.json", "w"), indent=1)
print("ranked", json.load(open("runs/screen.json"))["ranked"], "base means", a["base_mean"], b["base_mean"])
