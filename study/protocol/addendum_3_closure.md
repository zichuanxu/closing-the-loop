# Addendum 3 (2026-11-05): confirmation of the closure-contrast edit (CLC), frozen BEFORE any test-set run of CLC

## How CLC arose (exploratory, disclosed)

The pre-registered method C (CRD, layers 7/12 chosen by the teacher-forced screen) and the location-matched arm H
(CRD direction at Astra's layers) under-performed Astra. Exploratory dev work (`dev[64:192]`, 128 practice problems,
thinking, 32,768 tokens, 1 sample; no test data) then found:
- Astra's u, recovered exactly (rank-1 diff), has |cos| ≈ 0.71–0.74 with every automated contrast; a logit-lens read of
  its component orthogonal to CRD loads, after the α = 6 inversion, on closing tokens (`<|endoftext|>`, "accordingly",
  "results", "end").
- Hypothesis: the effective edit maps the loop state toward the state in which the model concludes. Automated
  direction CLC: `u_l = unit(mean recurrence-window state − mean state over the 64 tokens before the base model's own
  </think>)`, from the same 512 extract practice answers (256 recurrence windows; 43 finished answers). No correctness
  labels, no test data, no generation-based tuning of u.
- Three automated directions at Astra's location and strength (layers 15/19, α 6, 64 activation-ranked columns) on
  dev: CRD 39.1 %, BRC 53.1 %, **CLC 67.2 %** (cut 7.8 %), Astra 66.4 % (cut 14.8 %), base 13.3 %. CLC was chosen among
  these three on this dev set; location and strength were taken from Astra's published edit.

## Frozen edit (no further changes)

`edits/clcloc.npz` → `ckpt/clcloc` (sha256 recorded in `runs/frozen_clc.sha256` at freeze time).

## Confirmation arms (seed 0, 1 sample per problem, authors' protocols)

- I   CLC, MATH-500, thinking, 32,768 tokens — compared with A (base), B (Astra), H (CRD at the same location), all
      already run on the identical protocol.
- J   CLC, MATH-500, non-thinking, 8,192 tokens — compared with E (base) and F (Astra).
- K/L GSM8K test (1,319), thinking, 32,768 tokens: K = CLC, L = Astra (paired head-to-head; base not rerun, the
      authors report 13.8 % for it).

## Hypotheses (paired bootstrap over problems, 10,000 resamples)

- H7  I − A ≥ 20 points, CI lower bound > 0.
- H8  non-inferiority to Astra on MATH-500: I − B CI lower bound > −5 points.
- H9  non-inferiority to Astra on GSM8K: K − L CI lower bound > −5 points.
- H10 I − H > 0 with CI lower bound > 0 (the closure direction beats the plain recurrence direction at the same place).
- J vs E and F reported without a gate.

Known limits stated in advance: location (15, 19), α 6 and k 64 are borrowed from Astra, so CLC is an automated
*direction and column* rule, not a fully automated location search; one sample per problem; one checkpoint.
