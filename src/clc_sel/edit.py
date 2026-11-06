"""CLC-sel: a rank-one edit of an MLP down projection that stops verbatim reasoning loops.

Notation follows the paper. ``W`` is a down projection of shape ``(d, n)``: hidden size by MLP width.
``u`` is a unit direction in the residual stream, ``s = W.T @ u`` is what each neuron writes along ``u``,
and the edit is ``W' = W - alpha * u (m * s)^T`` with ``m`` a 0/1 mask over ``k`` neurons.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def _f64(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def unit(v: ArrayLike) -> Array:
    v = _f64(v)
    return v / np.linalg.norm(v)


def closure_direction(h_loop: ArrayLike, h_close: ArrayLike) -> Array:
    """Unit direction from the mean state before ``</think>`` to the mean state inside loops.

    Both inputs are window means of one layer's residual stream, shape ``(windows, d)`` or ``(d,)``.
    """
    loop, close = _f64(h_loop), _f64(h_close)
    if loop.ndim == 2:
        loop = loop.mean(0)
    if close.ndim == 2:
        close = close.mean(0)
    return unit(loop - close)


def select_neurons(
    W: ArrayLike, u: ArrayLike, a_rep: ArrayLike, a_first: ArrayLike, k: int = 64
) -> NDArray[np.int64]:
    """The ``k`` neurons whose write along ``u`` rises most when text repeats.

    Score of neuron ``c``: ``(a_rep[c] - a_first[c]) * s[c]``, with mean activations on the repeated text and
    on its first occurrence. Returns sorted indices.
    """
    s = _f64(W).T @ unit(u)
    score = (_f64(a_rep) - _f64(a_first)) * s
    return np.sort(np.argsort(-score)[:k])


def closed_form_alpha(
    W: ArrayLike,
    u: ArrayLike,
    cols: ArrayLike,
    a_rep: ArrayLike,
    a_pre: ArrayLike,
    gap: float,
    lo: float = 0.5,
    hi: float = 16.0,
) -> float:
    """Strength that moves the mean loop state by ``gap`` along ``u``, clipped to ``[lo, hi]``.

    ``gap`` is the distance between the mean loop state and the mean pre-closure state; ``a_pre`` is the mean
    activation before ``</think>``. Returns ``hi`` when the selected neurons do not separate the two states.
    """
    s = _f64(W).T @ unit(u)
    c = np.asarray(cols, dtype=np.int64)
    diff = float(s[c] @ (_f64(a_rep)[c] - _f64(a_pre)[c]))
    return float(np.clip(gap / diff, lo, hi)) if diff > 0 else hi


def apply_edit(W: ArrayLike, u: ArrayLike, cols: ArrayLike, alpha: float) -> Array:
    """``W - alpha * u (m * s)^T``: only the columns in ``cols`` change, each along ``u``."""
    W64, u64 = _f64(W), unit(u)
    s = W64.T @ u64
    m = np.zeros_like(s)
    m[np.asarray(cols, dtype=np.int64)] = 1.0
    return W64 - alpha * np.outer(u64, m * s)
