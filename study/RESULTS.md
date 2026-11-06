# Automated anti-loop weight edits vs. Weight-Space Programming (Qwen3.5-0.8B) — loopdir_001

Study folder on the GPU server: `experiments/loopdir_001/` (protocol, three addenda, code, logs, every run's full records).
All model work ran on A100s (GPU UUIDs `GPU-3a7e130c-…`, `GPU-40333bf8-…`, recorded per run in `summary.json`),
image `xu_loopdir:20261104` (`sha256:95af2a46…`, vLLM 0.30.0, Torch 2.13, Transformers 5.17, math-verify 0.9.0) and
`…b` (+ flash-linear-attention 0.4.1). Model: `Qwen/Qwen3.5-0.8B` @2fc06364 (text tensors). Author logs:
`RikiyaT/lm_understand-logs` @4d09ba6d. Classification of claims: **reproduced** (the blog's numbers on our hardware),
**new** (this study's experiments), **author-reported** (the blog / its logs).

## 1. Result in one paragraph

The blog's Experiment 8 edit was reproduced exactly (bit-exact tensors; MATH-500 49.2 % vs. the authors' 48.1 %). A
fixed, gradient-free rule — **closure contrast (CLC)**: `u = unit(mean residual state in the model's own verbatim loops
− mean state in the 64 tokens before its own </think>)`, written into the 64 MLP neurons that carry it most on loops,
inverted (`W ← W − 6·u·(m ⊙ uᵀW)`) — built from 512 base generations on practice problems in ~84 min of shared A100
time, with no LLM agent, scored **53.0 %** on MATH-500 (thinking): +45.6 points over base, +3.8 over Astra's 20-hour
agent edit (non-inferior; CI −0.2 to +7.8), and the only configuration in the study that significantly beats the
model's default non-thinking mode (+7.2, CI 2.8 to 11.8). It reaches the result with shorter answers (18.1k vs 21.7k
mean tokens) and fewer truncations (21.8 % vs 32.2 %). On GSM8K it is 3.8 points below Astra (CI −6.1 to −1.5; the
non-inferiority gate fails) with 35 % fewer tokens — it closes earlier and is less careful per answer. The
pre-registered first method (CRD with screened layers) failed its gates; CLC came out of the diagnosis of that failure
and was confirmed under a separately frozen addendum. Gates: H3, H7, H8, H10 pass; H1, H2, H5, H9 fail.

## 2. MATH-500 (500 problems, 1 sample, seed 0, authors' protocols; paired bootstrap, 10,000 resamples)

| Arm | Mode | Acc | Cut at cap | Mean tokens |
|---|---|---:|---:|---:|
| A base | thinking 32k | 7.4 | 93.0 | 30,867 |
| B Astra p50c (rebuilt bit-exact) | thinking | 49.2 | 32.2 | 21,742 |
| C CRD, screened layers 7/12, α 6 (pre-registered) | thinking | 16.4 | 84.2 | 29,891 |
| D random direction, same form | thinking | 8.6 | 91.6 | 30,508 |
| H CRD direction at Astra's layers 15/19 | thinking | 23.2 | 77.2 | 29,207 |
| **I CLC at 15/19 (addendum 3)** | thinking | **53.0** | **21.8** | 18,068 |
| E base | non-thinking 8k (model-card default) | 45.8 | 31.0 | 3,646 |
| F Astra | non-thinking | 44.4 | 31.6 | 3,711 |
| G CRD | non-thinking | 41.0 | 34.0 | 3,726 |
| J CLC | non-thinking | 43.0 | 27.8 | 3,500 |

Authors' own base / Astra (4 samples): 7.9 / 48.1 %, cut 93.0 / 33.1 % — reproduced within sampling noise.

| Pre-registered comparison | Δ points [95 % CI] | W/L | Gate |
|---|---|---|---|
| C − A (H1, ≥ 20) | +9.0 [6.0, 12.0] | 54/9 | fail |
| C − B (H2, lo > −5) | −32.8 [−37.0, −28.4] | 5/169 | fail |
| D − A (H3, < 5) | +1.2 [−1.4, 3.8] | 25/19 | pass |
| G − E (H5, ≥ 3) | −4.8 [−9.2, −0.2] | 54/78 | fail |
| I − A (H7, ≥ 20) | +45.6 [41.0, 50.2] | 231/3 | pass |
| I − B (H8, lo > −5) | +3.8 [−0.2, 7.8] | 61/42 | pass |
| I − H (H10, lo > 0) | +29.8 [25.6, 34.0] | 155/6 | pass |
| I − E (reported) | +7.2 [2.8, 11.8] | 85/49 | — |
| B − E (reported) | +3.4 [−0.8, 7.8] | 69/52 | — |

GSM8K test (1,319, thinking 32k, 1 sample; authors report base 13.8 %):

| Arm | Acc | Cut | Mean tokens | Acc on finished | Finished without box |
|---|---:|---:|---:|---:|---:|
| K CLC | 64.6 | 0.5 | 7,610 | 64.9 | 19 |
| L Astra | 68.4 | 5.4 | 11,738 | 72.3 | 53 |

K − L = **−3.8 [−6.1, −1.5]**, W/L 97/147: **H9 fails** — CLC is significantly worse than Astra on GSM8K, while using
35 % fewer tokens and almost never truncating. Together with the lower accuracy on finished MATH-500 answers
(67.3 vs 72.3) this points to over-early closure: CLC trades some per-answer care for termination. Parity with Astra
therefore holds on MATH-500 only.

Independent audit (separate verifier pass, `audit_independent/REPORT.md`): all 10 MATH-500 arms re-graded from raw
text with 0 disagreements; bootstrap CIs reproduced; every freeze timestamp precedes its arm's launch; frozen hashes
match; no extract/dev overlap with MATH-500.

## 3. What was learned (mechanism, new)

- **Astra's edit is exactly rank one** per matrix (σ₂/σ₁ ≈ 6×10⁻⁴); `W0 − 6·u·(m ⊙ uᵀW0)` with its recovered u and
  columns rebuilds 99.8 % of its bf16 entries bit-exactly. So the agent's program is fully described by one direction
  and 64 neuron indices per layer.
- Every automated "repetition" contrast — first occurrence vs. verbatim recurrence (CRD), matched-prefix repeating vs.
  diverging branches (BRC), loop vs. pre-closure (CLC) — has |cos| 0.71–0.74 with Astra's direction; CRD and BRC
  agree with each other at 0.90. Cosine does not predict function: at the same location, strength and column rule,
  dev accuracy was CRD 39.1, BRC 53.1, CLC 67.2 vs Astra 66.4 (base 13.3).
- The logit-lens read that led to CLC: Astra's component orthogonal to the recurrence direction, after the α = 6
  inversion, promotes closing tokens (`<|endoftext|>`, "accordingly", "results", "end"). The working edit does not just
  erase "I am repeating"; it pushes the loop state toward the state in which the model concludes.
- Column choice matters as much as direction (dev, Astra location): Astra u + automated columns 59.4; automated CRD u
  + Astra columns 50.8. Editing all columns instead of 64 (column-free `(I − αuuᵀ)W`, α 1–2) was weak (14.8–25.0).
- The pre-registered teacher-forced layer screen (largest drop on held-out recurrence log-prob, finished-text guard
  ≥ −0.01 nats/token) chose layers 7/12 and excluded 15/19 for their guard cost; generation contradicted it. Log-prob
  selectivity is a poor proxy for loop repair under sampling.
- Loop edits fix **verbatim** thinking loops (53/55 truncated base thinking answers end in an 8-gram-repeated tail) but
  not the default mode's truncations, which are mostly non-converging re-attempts (16/155 verbatim). No edit improved
  non-thinking accuracy (F −1.4, G −4.8, J −2.8 vs E).

## 4. Practical reading

- If thinking mode is used, the CLC edit is a cheap, reproducible fix: 131,072 changed weights (2 × 64 × 1,024), one
  bf16 checkpoint, no inference-time machinery, built in ~84 min of A100 time from the model's own outputs.
- The default non-thinking mode already reaches 45.8 % at ~1/5 of the tokens; CLC-thinking buys +7.2 points for
  ~5× tokens. Astra's edit did not significantly beat the default mode on this benchmark.

## 5. Next experiments (not run)

1. Fully automated location and strength: search layers × α with the CLC direction on dev generations (the
   teacher-forced screen is shown to be a bad proxy), then confirm on a fresh benchmark.
2. The closure trade-off: a smaller α or a mixed direction (CLC + recurrence) to recover GSM8K per-answer accuracy while
   keeping termination; measure accuracy on finished answers as the guard.
3. Transfer of the CLC rule to another looping checkpoint (e.g. Qwen3-0.6B thinking) without borrowing any agent edit.
4. Inherited, separate line: `experiments/native_strength_selection_001` acquisition completed cleanly (3,228 s of
   7,200 s, exit 0); its certification, selection and unsealing steps remain untouched.

## 6. Limits (stated, not resolved)

CLC borrows Astra's location (layers 15/19), strength (α 6) and sparsity (64 columns); only the direction and column
rule are automated, and CLC was picked among three directions on a 128-problem dev set before confirmation. One
sample per problem; one checkpoint; one benchmark family plus GSM8K. Gains come from ending loops (the blog's own
caveat applies): accuracy on finished answers is 67.3 % for CLC vs 72.3 % for Astra. Nearest prior on neuron-level loop
repair: Lazaridis et al. 2026 (arXiv 2606.13705, Gemma 4, ablation/attribution search); difference-of-means directions
and weight orthogonalization are standard (e.g. Arditi et al. 2024); the closure-state target and the head-to-head
with an agent-written edit are this study's contribution.

## 7. Engineering incidents (tooling, not model behavior)

Five vLLM engine starts failed on co-tenant memory changes during profiling (logs kept; relaunched; a retrying launcher
was added); one HF extraction OOM (batching reduced). The layer screen was split over two GPUs (same rule). All
deviations: `logs/deviations.txt`; freeze hashes: `protocol_freeze.sha256`, `runs/frozen_crd.sha256`,
`runs/frozen_clc.sha256`.

## 8. Addenda 4–5 (2026-11-05): CLC-sel, the early-closure fix, beats Astra on MATH-500 (new)

**Edit.** CLC-sel keeps CLC's direction, layers 15/19, α 6 and 64 columns, and changes only the column rule: neurons are
ranked by `(act_rep − act_first) · uᵀW_c`, the rise of their write along u when the same text is repeated versus its
first occurrence. The edit then acts mainly in loops. Built from the same 512 extract answers; no labels, no gradients.

**Selection (addendum 4, frozen 22:04:39Z before any run).** Fresh reserve set (256: 128 GSM8K-train, 128 MATH-train),
thinking, seed 0. Accuracy: Astra 61.3, CLC 60.9, CLC α 4 60.9, CLC α 3 57.8, **CLC-sel 66.4**, mixed direction 58.6.
Rule: highest reserve accuracy, must beat CLC → CLC-sel.

**Confirmation (seed 0, 1 sample).**

| | MATH-500 | cut | GSM8K test | cut | GSM8K tokens |
|---|---:|---:|---:|---:|---:|
| Astra (B / L) | 49.2 | 32.2 | 68.4 | 5.4 | 11,738 |
| CLC (I / K) | 53.0 | 21.8 | 64.6 | 0.5 | 7,610 |
| **CLC-sel (I2 / K2)** | **55.8** | **17.2** | **69.0** | 1.1 | 9,336 |

| Comparison | Δ points [95 % CI] | W/L | Gate |
|---|---|---|---|
| K2 − L (H11, lo > −5) | +0.6 [−1.8, 3.0] | 133/125 | pass (non-inferior; not superior) |
| K2 − K (H12, lo > 0) | +4.4 [2.0, 6.8] | 162/104 | pass |
| I2 − B (H13, lo > −5; superiority lo > 0) | **+6.6 [2.8, 10.4]** | 64/31 | pass; **superior** |
| I2 − I | +2.8 [−1.0, 6.6] | 55/41 | reported |
| I2 − E (default non-thinking) | +10.0 [5.4, 14.6] | 94/44 | reported |

**Seed replication (addendum 5, frozen 01:19:59Z).** MATH-500, seeds 1 and 2 for both arms; unit = problem (mean over
3 seeds). Per seed I2 − B: +6.6 [2.8, 10.4], +8.2 [4.4, 12.0], +5.2 [1.6, 8.8]. Pooled: CLC-sel 56.3 vs Astra 49.7,
**+6.7 [4.5, 8.9]**; H14 passes. Cut share per seed: CLC-sel 17.2/17.8/18.2 %, Astra 32.2/32.6/31.4 %.

**Independent audit** (`audit_addendum4/REPORT.md`): freeze order, selection, re-grading (0 disagreements on I2, K2 and
all comparison arms), bootstrap (bit-identical), leakage (0 id, text and near-duplicate overlaps) and the edit (exactly
two tensors, 64 columns each, rank 1, A100 `GPU-40333bf8-…`) all pass. The seed-replication arms were not part of
that audit; they use the same pipeline and grader.

**What it does and does not show.**
- On this model and protocol, a fixed gradient-free rule built in ~84 min of A100 time from the model's own outputs
  beats the blog's 20-hour agent-written edit on MATH-500 by 6.7 points (3 seeds) and matches it on GSM8K.
- The mechanism is fewer truncations, not more careful answers: finished-answer accuracy is 67.1 % (CLC-sel) vs 67.3 %
  (CLC) vs 72.3 % (Astra) on MATH-500, and 69.7 % vs 72.3 % (Astra) on GSM8K. The addendum's stated motive (raise
  finished-answer accuracy) is not supported; the selective columns end loops while ending fewer normal answers early.
- MATH-500 and GSM8K test were also used to confirm CLC; this is a second confirmation of a new edit on them. Layers,
  α and the 64-column budget are still Astra's. One checkpoint. Non-thinking mode was not retested with CLC-sel.
