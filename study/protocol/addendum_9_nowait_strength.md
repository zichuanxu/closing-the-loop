# Addendum 9 (2026-11-05): NoWait baseline and a half-strength automatic screen, frozen BEFORE any run below

Motivation, stated plainly: the addendum-7 automatic edit (layer 13, α* = 11.8) tied Astra on MATH-500 (H18 fail,
+0.4 [−3.4, 4.2]) with 10.2% truncation but 55.0% finished-answer accuracy against Astra's 72.3%, which suggests the
strength overshoots. This addendum was written after seeing those test results, so any new test-set confirmation is
the third use of each test set and will be reported as such. The goal is a published decoding baseline (NoWait) and
one pre-specified strength variant chosen on the screening set only.

## Part A: NoWait (Wang et al., Findings of EMNLP 2025)

- Keywords (the paper's list, 17 words): wait, alternatively, hmm, but, however, alternative, another, check,
  double-check, oh, maybe, verify, other, again, now, ah, any.
- Suppressed tokens: every vocabulary id whose decoded text, lowercased and with every character outside [a-z-]
  removed, equals a keyword. This covers " wait", "Wait", ".wait", "WAIT". Deviation: the paper matches substrings and
  then prunes false matches by hand; hand pruning cannot be reproduced, so we use exact word match.
- Mechanism: vLLM logit_bias −100 on those ids at every step (thinking and answer), protocol think35 otherwise
  unchanged, 32,768 tokens, seed 0. The run stores the suppressed ids and counts any that still appear.
- Runs: m500_N_nowait (base + NoWait, MATH-500), gsm_N_nowait (base + NoWait, GSM8K test),
  m500_N2_clcsel_nowait (CLC-sel + NoWait, MATH-500).
- H21: I2 − m500_N_nowait, CI lower bound > 0.  H22: K2 − gsm_N_nowait, CI lower bound > 0.
- Reported: m500_N_nowait − A (base), m500_N2_clcsel_nowait − I2.

## Part B: half-strength automatic screen (screening set only)

- Stage D: single-layer CLC-sel edits at α*_l / 2 for l ∈ {9, 11, 13, 15, 17, 19} (the layers with ≥ 50% in stage B),
  on data/screen.jsonl, think35, 32,768 tokens, seed 0, same as stage B.
- Pick v2: highest screen accuracy over stage B (12 singles at α*), stage C (the pair) and stage D (6 singles at α*/2).
  Ties → higher finished-answer accuracy, then smaller total α, then lower layer.
- If pick v2 is the addendum-7 pick (layer 13 at α*), H18/H19 stand and no new test run is made.
- Otherwise confirm on MATH-500 (m500_I4_auto2) and GSM8K test (gsm_K4_auto2), seed 0.
- H23: I4 − B (Astra), CI lower bound > 0.  H24: K4 − L (Astra), CI lower bound > −5.
- Reported: I4 − I2, K4 − K2, I4 − I3.

Statistics as before: paired bootstrap over problems, 10,000 resamples, seed 0, 95% percentile intervals.

## Status

Stopped by the user at 2026-11-05T12:20:01Z before any GPU run started. Only the stage-D edits (edits/auto_h*.npz, ckpt/auto_h*) were built. H21-H24 were not tested.
