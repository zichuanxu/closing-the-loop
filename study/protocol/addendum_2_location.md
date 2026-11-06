# Addendum 2 (2026-11-04, after the layer screen, BEFORE any dev-set or MATH-500 result of an automated edit)

The frozen screen selected layers 7 and 12 (`runs/screen.json`). Astra's layers 15 and 19 had large recurrence drops
(−0.062 / −0.059 nats/token) but failed the finished-text guard (−0.020 / −0.028 < −0.01). To separate the automated
*direction* from the automated *location*, one arm is added, with every setting fixed now:

- H  CRD-at-Astra-location: u from `runs/extract/directions.npz` (key `all`) at layers 15 and 19, α 6, k 64,
     activation-ranked columns (the CRD column rule) — Astra's location and strength with the automated direction.
     MATH-500, thinking protocol, 32,768 tokens, 1 sample, seed 0. No tuning.

Added comparison (descriptive, paired bootstrap): H − B (automated vs Astra direction at the same place and strength),
H − A, and H − C (location choice). H does not replace C as the method's pre-specified output.
