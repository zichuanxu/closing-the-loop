import json
from pathlib import Path

import numpy as np

from clc_sel import apply_edit, closed_form_alpha, closure_direction, select_neurons, unit
from clc_sel.apply import load_spec

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
d, n, k = 16, 40, 5
W = rng.standard_normal((d, n))
u = unit(rng.standard_normal(d))


def test_edit_is_rank_one_on_selected_columns():
    cols = np.array([1, 7, 30])
    D = apply_edit(W, u, cols, 6.0) - W
    assert np.linalg.matrix_rank(D) == 1
    assert np.flatnonzero(np.abs(D).sum(0) > 0).tolist() == cols.tolist()
    # selected columns lose (alpha) times their component along u; alpha 1 removes it
    assert np.allclose(u @ apply_edit(W, u, cols, 1.0)[:, cols], 0)
    assert np.allclose(u @ apply_edit(W, u, cols, 6.0)[:, cols], -5 * (u @ W[:, cols]))


def test_select_neurons_ranks_loop_rise_times_write():
    a_first = rng.random(n)
    a_rep = a_first.copy()
    s = W.T @ u
    top = np.argsort(-np.abs(s))[:k]
    a_rep[top] += np.sign(s[top]) * 10  # these neurons write much more along u on repeats
    assert set(select_neurons(W, u, a_rep, a_first, k)) == set(top)


def test_closure_direction_and_alpha():
    loop, close = rng.standard_normal((8, d)) + 3 * u, rng.standard_normal((8, d))
    v = closure_direction(loop, close)
    assert np.isclose(np.linalg.norm(v), 1) and v @ u > 0.8
    cols = np.arange(k)
    a_pre = np.zeros(n)
    a_rep = np.zeros(n)
    a_rep[cols] = 1.0 / (W.T @ u)[cols] / k  # sum_c s_c (a_rep - a_pre) = 1
    assert np.isclose(closed_form_alpha(W, u, cols, a_rep, a_pre, gap=4.0), 4.0)
    assert closed_form_alpha(W, u, cols, a_rep, a_pre, gap=100.0) == 16.0
    assert closed_form_alpha(W, u, cols, -a_rep, a_pre, gap=4.0) == 16.0


def test_released_specs_load():
    spec = load_spec(ROOT / "edits/clc_sel.json")
    assert sorted(spec) == [f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15, 19)]
    for e in spec.values():
        assert len(e["cols"]) == 64 and len(e["u"]) == 1024 and e["alpha"] == 6.0
    auto = load_spec(ROOT / "edits/auto_layer13.json")
    assert [round(e["alpha"], 2) for e in auto.values()] == [11.82]
    assert json.loads((ROOT / "edits/clc.json").read_text())["args"]["alpha"] == 6.0
