# Addendum 8 (2026-11-05): seed replication of CLC-sel vs Astra on GSM8K, frozen BEFORE any run below

Addendum 4 found CLC-sel (K2) non-inferior to Astra (L) on GSM8K test with one sample per problem (seed 0):
+0.6 [−1.8, 3.0]. This adds sampling seeds 1 and 2 for both arms; no edit changes.

Arms (GSM8K test, 1,319 problems, think35, 32,768 tokens, 1 sample each):
- L_s1, L_s2   Astra (`ckpt/astra_p50c`), seeds 1 and 2.
- K2_s1, K2_s2 CLC-sel (`ckpt/clc_sel`), seeds 1 and 2.

Analysis (same as addendum 5): per problem, mean correctness over seeds {0, 1, 2}; unit = problem; paired bootstrap
over problems, 10,000 resamples, seed 0.
- H20 pooled K2 − L, CI lower bound > −5 (non-inferiority holds across seeds; same margin as H9/H11).
- Reported: superiority (CI lower bound > 0); each seed separately; cut share and finished-answer accuracy per seed.
