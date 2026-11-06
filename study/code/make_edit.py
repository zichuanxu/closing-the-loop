"""Compile directions into Astra-form MLP writer edits:  W = W0 - alpha * outer(u, mask_k * (u @ W0)).

u = unit direction at the residual output of layer l; W0 = that layer's mlp.down_proj (hidden x intermediate);
mask_k keeps the k intermediate columns with the largest |u @ W0| (k = 0 means all columns).

  python make_edit.py --dirs directions.npz [--key all] --layers 15 19 --alpha 6 --k 64 --out edit.npz
  python make_edit.py --astra-u /work/ckpt/astra_p50c --layers 15 19 ...  (u recovered from Astra's released diff)
"""
import argparse, json

import numpy as np
from safetensors.torch import load_file

RELEASE = "/models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"


def load_f32(path, names):
    sd = load_file(path)
    return {n: sd[n].float().numpy() for n in names}


def edit(W0, u, alpha, k, act=None, cols=None):
    """act: mean neuron activation on recurrence windows; then columns are ranked by their measured write along +u
    (act_c * u.W_c, largest first), otherwise by |u.W_c|. cols: a fixed column set instead (diagnostics)."""
    s = u @ W0
    m = np.ones_like(s)
    if cols is not None:
        m = np.zeros_like(s)
        m[cols] = 1
    elif k:
        score = act * s if act is not None else np.abs(s)
        m = np.zeros_like(s)
        m[np.argsort(-score)[:k]] = 1
    return W0 - alpha * np.outer(u, m * s), np.flatnonzero(m)


def recover_u(d, W0):
    """Astra's edit d = -alpha u (m * u W0): u = top left singular vector, signed so that u . (d (W0^T u)) < 0."""
    # ponytail: power iteration on the 1024x1024 Gram matrix (full SVD hung under host CPU contention)
    G = d @ d.T
    u = np.ones(G.shape[0])
    for _ in range(200):
        u = G @ u
        u /= np.linalg.norm(u)
    s1 = u @ G @ u
    R = G - s1 * np.outer(u, u)            # second eigenvalue: rank-1 quality of the edit
    v = np.random.default_rng(0).standard_normal(G.shape[0])
    for _ in range(200):
        v = R @ v
        v /= np.linalg.norm(v)
    if u @ d @ (W0.T @ u) > 0:
        u = -u
    return u, float(np.sqrt(max(v @ R @ v, 0) / s1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs")
    ap.add_argument("--key", default="all")
    ap.add_argument("--astra-u", help="Astra-edited ckpt dir (source for --u-from / --cols-from astra)")
    ap.add_argument("--u-from", choices=("dirs", "astra"), default="astra")
    ap.add_argument("--cols-from", choices=("rule", "astra"), default="rule")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", help="per-layer alpha (addendum 7); overrides --alpha")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--select", choices=("act", "weight"), default="act")
    ap.add_argument("--random-seed", type=int, help="replace u by a random unit vector (norm-matched control)")
    ap.add_argument("--mix", help="second directions.npz; u = unit(u + u_mix)")
    ap.add_argument("--act-first", help="directions.npz holding act_first; columns ranked by act_rep - act_first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = [NAME.format(l) for l in a.layers]
    W0 = load_f32(RELEASE, names)
    U, info, ACT, FORCE = {}, {}, {}, {}
    Z = np.load(a.dirs) if a.dirs else None
    if Z is not None:
        U = {l: Z[a.key][l] for l in a.layers}
        if a.mix:  # addendum 4: u = unit(u + u_mix), u_mix sign-aligned to u
            ZM = np.load(a.mix)["all"]
            U = {l: U[l] / np.linalg.norm(U[l]) + np.sign(U[l] @ ZM[l]) * ZM[l] / np.linalg.norm(ZM[l]) for l in a.layers}
        if a.select == "act":
            ACT = {l: Z["act_rep"][l].astype(np.float64) for l in a.layers}
            if a.act_first:  # addendum 4: loop-selective columns, rank by (act_rep - act_first) * u.W_c
                AF = np.load(a.act_first)["act_first"]
                ACT = {l: ACT[l] - AF[l].astype(np.float64) for l in a.layers}
    if a.astra_u:  # diagnostics: Astra's u (--u-from astra) and/or Astra's columns (--cols-from astra)
        WA = load_f32(f"{a.astra_u}/model.safetensors", names)
        for l, n in zip(a.layers, names):
            d = WA[n].astype(np.float64) - W0[n]
            ua, info[n] = recover_u(d, W0[n].astype(np.float64))
            if Z is not None and ua @ Z[a.key][l] < 0:
                ua = -ua  # orient like the CRD u, so the activation column rule reads the same sign
            if a.u_from == "astra":
                U[l] = ua
            if a.cols_from == "astra":
                FORCE[l] = np.flatnonzero(np.abs(d).sum(0) > 0)
    rng = np.random.default_rng(a.random_seed) if a.random_seed is not None else None
    out, meta = {}, {}
    alphas = dict(zip(a.layers, a.alphas)) if a.alphas else {l: a.alpha for l in a.layers}
    for l, n in zip(a.layers, names):
        u = U[l].astype(np.float64)
        if rng is not None:
            u = rng.standard_normal(u.shape)
        u = u / np.linalg.norm(u)
        W, cols = edit(W0[n].astype(np.float64), u, alphas[l], a.k, ACT.get(l) if rng is None else None, FORCE.get(l))
        out[n] = W.astype(np.float32)
        meta[n] = {"cols": cols.tolist(), "u": u.tolist(), "astra_sv_ratio": info.get(n),
                   "rel_change": float(np.linalg.norm(W - W0[n]) / np.linalg.norm(W0[n]))}
    np.savez(a.out, **out)
    json.dump({"args": vars(a), "tensors": meta}, open(a.out.replace(".npz", ".json"), "w"))
    print({n: (m["rel_change"], m["astra_sv_ratio"]) for n, m in meta.items()})


if __name__ == "__main__":
    main()
