# Addendum 6 (2026-11-05): CLC-sel vs published loop/termination methods, frozen BEFORE any run below

Goal: compare CLC-sel (addendum 4) not only with the blog's edits but with published inference-time methods that target
the same failure (thinking that loops until the token cap), at a matched total token budget of 32,768 per answer.

## Methods

- **Repetition penalty** (Keskar et al. 2019, CTRL; vLLM `repetition_penalty`), on the base model, added on top of the
  protocol's sampling (presence penalty 1.5 stays). Strength chosen on the reserve set among {1.05, 1.1, 1.2} (highest
  accuracy; tie → smaller); the pick runs on MATH-500 as arm P.
- **Budget forcing** (Muennighoff et al. 2025, s1): thinking is generated up to 30,720 tokens; any answer that reaches
  that cap without `</think>` gets `\n</think>\n\n` appended (token ids) and continues for up to 2,048 more tokens with
  the same sampling; total ≤ 32,768. The 30,720 / 2,048 split is fixed here, not tuned. Arms on MATH-500: A_bf (base),
  B_bf (Astra p50c), I2_bf (CLC-sel).
- **Blog edits** (author-reported, MATH-500, 4 samples): p10c 12.4 %, p25c 14.1 %, p50c 48.1 % (p50c reproduced here
  as B, 49.2 %).
- Already run: E (default non-thinking, 45.8 %), B (Astra), I2 (CLC-sel), A (base).

All arms: think35 sampling, seed 0, 1 sample, authors' grader.

## Hypotheses (paired bootstrap over 500 problems, 10,000 resamples)

- H15 I2 − P, CI lower bound > 0 (CLC-sel beats the tuned repetition penalty).
- H16 I2 − A_bf, CI lower bound > 0 (CLC-sel alone beats budget forcing on the base model).
- H17 I2_bf − B_bf, CI lower bound > 0 (the edge over Astra survives when both get budget forcing, i.e. it is not only
  from fewer truncations).
- Reported: I2_bf − I2, A_bf − A, B_bf − B, I2_bf − E; finished-answer accuracy and forced share per arm.

A failed H17 would mean CLC-sel's advantage over Astra is mostly termination and is reported as such.
