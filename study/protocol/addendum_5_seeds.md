# Addendum 5 (2026-11-05): seed replication of CLC-sel vs Astra on MATH-500, frozen BEFORE any run below

Addendum 4 confirmed CLC-sel (I2) over Astra (B) on MATH-500 with one sample per problem (seed 0). A single sample per
problem leaves sampling noise in each arm. This addendum adds two more sampling seeds per arm; no edit changes.

Arms (MATH-500, think35, 32,768 tokens, 1 sample each):
- B_s1, B_s2   Astra (`ckpt/astra_p50c`), seeds 1 and 2.
- I2_s1, I2_s2 CLC-sel (`ckpt/clc_sel`), seeds 1 and 2.

Analysis: per problem, mean correctness over seeds {0, 1, 2} for each arm; the unit is the problem (500), not the
sample. Paired bootstrap over problems (10,000 resamples, seed 0).
- H14 mean(I2) − mean(B) > 0 with CI lower bound > 0 (superiority holds when sampling noise is averaged).
- Reported: each seed's I2 − B separately; per-arm mean accuracy and cut share per seed.
