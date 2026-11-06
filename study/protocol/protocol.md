# loopdir_001 — automated recurrence-direction weight edits vs. Weight-Space Programming (frozen 2026-11-04, before any automated edit exists)

## Question

The Weight-Space Programming post (Takehi et al., 2026; logs `RikiyaT/lm_understand-logs` @4d09ba6d) reports that an LLM
agent (Astra) reading Qwen3.5-0.8B's weights, with 50 forward passes over 20.3 h, wrote a static edit of 131,072 weights
(64 MLP writer columns in layers 15 and 19, `W = W0 - 6 u (mask ⊙ uᵀW0)`) that lifts thinking-mode MATH-500 from 7.9 % to
48.1 % by curing repetition loops. Its four runs with fewer passes or other prompts gained 0–6 points. The post has no
automated baseline. **Can a fixed, gradient-free, agent-free rule compile an equally good edit from the base model's own
outputs, and is the edit's direction the one Astra found?**

## Method under test (CRD: contrastive recurrence direction), fixed rule

1. Base model, thinking on, the authors' sampling (T 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0), one
   answer for each of the 512 `extract` practice problems (256 MATH train + 256 GSM8K train from the authors' practice
   files, which exclude evaluation problems), capped at 16,384 new tokens.
2. In each answer, the first 64-token block that recurs verbatim later without overlap. Teacher-forced residual states
   at every layer output, averaged over the first occurrence and over the recurrence. `u_l = unit(mean(recur − first))`
   over at most 256 such answers; MLP neuron activations on the recurrence windows also averaged.
3. Edit form identical to Astra's: `W_l = W0_l − α u_l (m ⊙ u_lᵀW0_l)` on `layers.l.mlp.down_proj`, `m` = the 64
   columns with the largest measured write along +u (`act_c · u_lᵀW_c` on recurrence windows). Same weight count as
   Astra (2 layers × 64 columns × 1,024).
4. Layer choice: teacher-forced screen, α 6, every layer singly; log-prob change on held-out recurrence windows (odd
   answers) and on 64-token windows inside finished base answers. Keep layers whose finished-text change ≥ −0.01
   nats/token; take the two with the largest recurrence-log-prob drop.
5. α ∈ {2, 4, 6, 8} for the two-layer edit, chosen by accuracy on `dev[64:192]` (128 practice problems; thinking
   protocol, 32,768 tokens, one sample); tie → smaller α. `dev[0:64]` was used for the pipeline smoke test only.
6. The chosen edit is frozen (tensor hashes recorded) before any MATH-500 generation of an edited model.

## Arms on MATH-500 (500 problems, the authors' thinking protocol, 32,768 tokens, 1 sample per problem, seed 0)

A base · B Astra-p50c (rebuilt bit-exact from the released diff) · C CRD (this method) · D random-direction control
(same layers, α, k, column rule replaced by |uᵀW|; u uniform random unit, seed 0) · E base in non-thinking mode (the
model card's default; authors' non-thinking sampling, 8,192 tokens) — a deployment reference, not an edit.

## Hypotheses and decision rules (paired by problem; 95 % bootstrap CI, 10,000 resamples over problems)

- H1 practical: C − A ≥ 20 points with CI lower bound > 0.
- H2 parity with Astra: C − B CI lower bound > −5 points (non-inferiority, margin 5).
- H3 specificity: D − A < 5 points (a random direction of the same form does not explain the gain).
- H4 mechanism (descriptive): cosine between u_l(CRD) and Astra's u (recovered: top singular vector of its rank-1 diff,
  σ₂/σ₁ < 1e−3) at layers 15 and 19; reported whatever the value.
- Cost: GPU-seconds of steps 1–5 vs. Astra's 20.3 h plus 50 generation/teacher-forced passes and a frontier LLM.
- E is reported against C without a gate: if non-thinking matches or beats C, the practical value of loop edits for
  this checkpoint is limited to users who need thinking mode, and this will be stated.

Reported regardless of outcome: accuracy, share cut at 32,768 tokens, mean tokens, answers closed without a box,
losses (base right → edited wrong). One sample per problem is noisier than the authors' 4; that limitation stands.
Units are problems; the same problem's arms are paired, never counted as independent.

## Not claimed

Capability addition (gains here are expected to come from ending loops — the post's own caveat); novelty of
difference-of-means directions or weight orthogonalization (Arditi et al. 2024 refusal direction; activation steering);
generality beyond Qwen3.5-0.8B until tested.
