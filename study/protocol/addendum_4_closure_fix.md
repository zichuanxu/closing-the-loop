# Addendum 4 (2026-11-05): fixing CLC's early closure, frozen BEFORE any run below

## Problem (from confirmed results, addendum 3)

CLC (I/K) beats base and matches Astra on MATH-500 but is 3.8 points below Astra on GSM8K (CI −6.1 to −1.5) with 35 %
fewer tokens, and its accuracy on finished answers is lower (MATH-500 67.3 vs 72.3). Dev[64:192] shows the same split:
finished-answer accuracy CRD 0.907 > BRC 0.829 > Astra 0.771 > CLC 0.729, while cut share runs the other way. The
closure push also fires during ordinary reasoning, so answers end before they are checked.

## Candidates (all at layers 15/19, 64 columns, built from the same 512 extract answers; no new data, no labels)

- R3 `clc_a4`   CLC direction, α 4 (weaker push).
- R4 `clc_a3`   CLC direction, α 3.
- R5 `clc_sel`  CLC direction, α 6, loop-selective columns: rank by `(act_rep − act_first) · uᵀW_c` (neurons whose
                write along u rises when the same text is repeated, so the edit fires less on first-pass reasoning).
- R6 `mix`      `u = unit(û_clc + û_crd)`, α 6, act_rep columns (halfway between closing and anti-repeat).
References on the same set: R0 CLC (α 6, as frozen in addendum 3), R1 Astra.

## Selection set (fresh)

`data/reserve.jsonl`, all 256 problems (128 GSM8K-train, 128 MATH-train; drawn in protocol.md, never used so far),
thinking, 32,768 tokens, seed 0, 1 sample.

Rule: pick the candidate with the highest overall reserve accuracy (ties: higher finished-answer accuracy). It goes to
confirmation only if it beats R0 on reserve accuracy; otherwise the result is reported as negative.

## Confirmation (pick only; seed 0, 1 sample, authors' think35 protocol)

- I2  pick on MATH-500 (500), compared with B (Astra) and I (CLC).
- K2  pick on GSM8K test (1,319), compared with L (Astra) and K (CLC).

Hypotheses (paired bootstrap over problems, 10,000 resamples):
- H11 K2 − L CI lower bound > −5 (GSM8K non-inferiority to Astra, the gate CLC failed).
- H12 K2 − K > 0 with CI lower bound > 0 (the fix helps on GSM8K).
- H13 I2 − B CI lower bound > −5 (MATH-500 non-inferiority to Astra kept).
- Reported: I2 − B and K2 − L as superiority tests (CI lower bound > 0); I2 − I; finished-answer accuracy.

Limits stated in advance: MATH-500 and GSM8K test were already used to confirm CLC; this is a second confirmation of a
new edit on the same benchmarks, and location/strength family are still Astra's.
