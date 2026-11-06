# Addendum 7 (2026-11-05): automated layer and strength for CLC-sel, frozen BEFORE α* is computed or any run below

CLC-sel (addendum 4) borrowed layers 15/19 and α 6 from Astra's edit. This addendum replaces both with a fixed rule that
uses no information from Astra's edit. The 64-column budget is kept (stated limit).

## Closed-form strength α*_l (no generation)

At layer l (residual output), from the same 512 extract answers: h̄_rep = mean state over the 256 verbatim-recurrence
windows (as before); h̄_pre and the down_proj input means ā_pre over the 64 tokens before `</think>` in the finished
answers (recomputed by `code/closure_stats.py`, same windows as `closure_dirs.py`). u_l = unit(h̄_rep − h̄_pre), gap
g_l = ‖h̄_rep − h̄_pre‖. Columns m_l = CLC-sel rule at layer l (top 64 by (ā_rep − ā_first)·uᵀW_c). The edit
`W ← W − α u (m ⊙ uᵀW)` lowers the layer's write along u by α·Σ_{c∈m} s_c a_c, s = uᵀW. Choose α so that the
loop/pre-closure gap along u is closed by the edit's differential effect:

    α*_l = g_l / Σ_{c∈m_l} s_c (ā_rep,c − ā_pre,c),   clipped to [0.5, 16].

## Layer choice (generation screen)

- Screen set: `data/screen.jsonl` = dev[0:64] + dev[192:256] (128 practice problems; dev[0:64] was used only for the
  base/Astra smoke tests, dev[192:256] never), thinking, 32,768 tokens, seed 0.
- Stage B: single-layer CLC-sel edits at α*_l for l ∈ {1, 3, 5, …, 23} (12 layers).
- Stage C: the two best single layers by screen accuracy (ties → higher finished-answer accuracy, then lower layer),
  combined in one edit with α*_l / 2 at each of the two layers (split the gap).
- Pick: stage-C pair if its screen accuracy is higher than the best single layer, otherwise the best single layer.
- If every stage-B layer scores below the base model's screen accuracy, the procedure has failed and is reported so.

## Confirmation (pick only; seed 0, 1 sample, think35)

- I3 on MATH-500, K3 on GSM8K test; compared with B/L (Astra) and I2/K2 (CLC-sel at Astra's location).
- H18 I3 − B, CI lower bound > 0 (the fully automated edit beats Astra on MATH-500).
- H19 K3 − L, CI lower bound > −5 (GSM8K non-inferiority to Astra).
- Reported: I3 − I2, K3 − K2; α* for every layer; whether the pick coincides with 15/19.

If the pick is exactly CLC-sel's configuration (layers 15/19 with α within ±0.5 of 6), the existing I2/K2 runs are its
confirmation and no new test run is needed; this is stated here in advance.
