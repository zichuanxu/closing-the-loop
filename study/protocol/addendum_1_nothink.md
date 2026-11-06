# Addendum 1 (2026-11-04, written after arm E finished and BEFORE any edited-model MATH-500 result was available)

Arm E (base, non-thinking, model-card default, 8,192 tokens) scored 45.8 % on MATH-500 with 31.0 % of answers cut at the
cap: non-thinking mode also runs out of budget. The practically relevant question is therefore whether a loop edit
improves the model's default configuration, not only the thinking configuration the post evaluated.

Added arms (same MATH-500, same non-thinking protocol, 8,192 tokens, 1 sample, seed 0):
- F  Astra-p50c, non-thinking
- G  CRD (the edit frozen by protocol step 6, unchanged; no re-selection for this mode), non-thinking

Added hypotheses (paired bootstrap as in the protocol):
- H5: G − E ≥ 3 points with 95 % CI lower bound > 0 (the automated edit improves the default configuration).
- H6 (descriptive): F − E, and the best edited arm over all modes vs. the best unedited arm over all modes.
Losses (E right → G wrong) and cut share are reported. No change to the original protocol's arms, gates or selection.
