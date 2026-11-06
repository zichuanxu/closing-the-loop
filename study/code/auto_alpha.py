"""Addendum 7: closed-form strength alpha*_l = g_l / sum_{c in m_l} s_c (a_rep,c - a_pre,c), clipped to [0.5, 16].

  python auto_alpha.py --clc runs/clc/directions.npz --extract runs/extract --stats runs/clc_stats/closure_stats.npz \
      --out runs/auto_alpha.json
"""
import argparse, json

import numpy as np
from safetensors.torch import load_file

RELEASE = "/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"


def alpha_star(u, W, a_rep, a_first, a_pre, gap, k=64):
    s = u @ W
    cols = np.argsort(-((a_rep - a_first) * s))[:k]          # CLC-sel columns
    diff = float(s[cols] @ (a_rep[cols] - a_pre[cols]))
    return float(np.clip(gap / diff, 0.5, 16)) if diff > 0 else 16.0, diff, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clc", required=True)
    ap.add_argument("--extract", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z, E, S = np.load(a.clc), np.load(f"{a.extract}/directions.npz"), np.load(a.stats)
    rep = np.load(f"{a.extract}/window_means.npz")["rep"].mean(0)   # [layers, H]
    sd = load_file(RELEASE)
    res = {}
    for l in range(rep.shape[0]):
        d = rep[l].astype(np.float64) - S["pre_resid"][l]
        gap = float(np.linalg.norm(d))
        u = Z["all"][l].astype(np.float64)
        assert abs(u @ d / gap - 1) < 1e-3, (l, u @ d / gap)    # same direction as closure_dirs
        W = sd[NAME.format(l)].float().numpy().astype(np.float64)
        al, diff, _ = alpha_star(u, W, E["act_rep"][l].astype(np.float64), E["act_first"][l].astype(np.float64),
                                 S["act_pre"][l].astype(np.float64), gap)
        res[l] = {"alpha_star": al, "gap": gap, "diff_write": diff}
        print(l, res[l], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
