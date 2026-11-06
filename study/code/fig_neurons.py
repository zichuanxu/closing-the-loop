"""Figure input (container, numpy): CLC-sel neuron scores r_c = (a_rep - a_first) * s_c at one layer, sorted.
  python code/fig_neurons.py --layer 19 --out runs/fig_neurons_L19.json
"""
import argparse, json

import numpy as np
from safetensors.torch import load_file

RELEASE = "/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=19)
ap.add_argument("--out", required=True)
a = ap.parse_args()
l = a.layer
Z, E = np.load("runs/clc/directions.npz"), np.load("runs/extract/directions.npz")
W = load_file(RELEASE)[f"model.language_model.layers.{l}.mlp.down_proj.weight"].float().numpy().astype(np.float64)
u = Z["all"][l].astype(np.float64)
s = u @ W
r = (E["act_rep"][l].astype(np.float64) - E["act_first"][l].astype(np.float64)) * s
plain = E["act_rep"][l].astype(np.float64) * s
top_sel, top_plain = np.argsort(-r)[:64], np.argsort(-plain)[:64]
order = np.argsort(-r)
json.dump({"layer": l, "n": int(r.size), "sorted_r": r[order][:256].tolist(), "r_min": float(r.min()),
           "overlap_sel_plain": int(len(set(top_sel) & set(top_plain))),
           "share_r_top64": float(r[top_sel].sum() / r[r > 0].sum())}, open(a.out, "w"))
print(l, "overlap CLC-sel vs CLC top-64:", len(set(top_sel) & set(top_plain)), "top64 share of positive r:",
      round(float(r[top_sel].sum() / r[r > 0].sum()), 3))
