# Independent audit of addendum 4 (closure fix), loopdir_001

Auditor run: 2026-11-05 ~01:20-01:45Z, read-only except this folder. Scripts: `audit_a4.py`, `audit_edit.py` (run in image `xu_loopdir:20261104b`, no GPU, no network, grader mounted read-only at /authors). Raw numbers: `audit.json`, `audit_part1.json`, `audit_part2.json`.

## Summary

| # | Check | Result |
|---|-------|--------|
| 1 | Timing / freeze | PASS (one thin margin, see notes) |
| 2 | Selection | PASS |
| 3 | Re-grading | PASS (0 disagreements) |
| 4 | Statistics / gates | PASS (bit-identical to analysis_a4.json) |
| 5 | Leakage | PASS (0 id, 0 text, 0 near-duplicate overlaps) |
| 6 | The edit and GPUs | PASS |

## 1. Timing

- `sha256sum -c runs/frozen_fix.sha256`: all 12 hashed files OK (addendum 4, code/make_edit.py, six edits npz, four ckpt safetensors). The 13th line is the timestamp, which sha256sum flags as improperly formatted (expected).
- Freeze timestamp (last line) `2026-11-04T22:04:39Z`; file mtime 07:04:39.55 JST = 22:04:39.55Z.
- Every ckpt/edit file hashed was written before that (clc_sel model 07:03:58 JST, mix 07:04:07 JST, clc_sel.npz 07:03:24 JST, make_edit.py 07:02:35 JST).
- First reserve launch, `res_R1_astra`: launches.txt 22:04:40Z; `docker inspect` Created 22:04:39.87Z, Started 22:04:40.35Z. So the freeze file precedes the first reserve container by about 0.3 s (create) / 0.8 s (start). All later res_* (22:08:41Z to 22:38:47Z) are minutes later. Ordering holds, margin on the first run is thin; it looks like the freeze and launch are chained in one script.
- Selection record `runs/selected_fix.txt` stamped `2026-11-04T23:05:41Z` (file mtime 08:05:47.98 JST = 23:05:47.98Z). First I2 launch 23:05:48Z. Reserve runs ended 22:34Z to 23:05:17Z (R6 container FinishedAt 23:05:17Z), so selection was made after all six reserve runs finished. Note the file mtime is 6 s later than the stamp inside it and only ~0.1 s before the I2 launch line; the container for the failed first I2 start was created ~23:05:48Z. Order holds.
- I2 was launched three times; the first two died in vLLM engine init (log: negative available KV cache memory, GPU shared with other jobs), leaving empty `*.failed_start_*` directories and no records. Third launch (23:50:57Z, container started 23:50:57.5Z) is the reported run. Same config each time, no outcome was seen before relaunch, so no selection effect. K2 launched 23:54:58Z.
- Not hash-frozen: `code/analyze.py` (mtime 08:05:57 JST = 23:05:57Z, i.e. written 9 s after the I2 launch and before any I2/K2 outcome). Its gate thresholds and bootstrap match the addendum text. `code/gen_eval.py` is in the older protocol_freeze.sha256 and still OK; `code/make_edit.py` fails the OLD protocol_freeze hash (edited for addendum 4 with --mix/--act-first) but matches the new frozen_fix hash. `edits/clcloc.npz` and `clc_recheck.npz` share one hash, so R0 reproduces from the modified make_edit.py.
- `data/SHA256SUMS` all OK (reserve.jsonl, extract.jsonl, math500.jsonl, gsm8k_test.jsonl, dev.jsonl); data files predate the freeze.
- Extra, not preregistered: `m500_I2_clcsel_s1` (seed 1) launched 2026-11-05T01:20:00Z and was still running at audit time. It is not part of H11-H13; if reported it must be labelled post hoc.

## 2. Selection (reserve, 256 problems, re-graded from raw text, 0 disagreements for all six)

| Arm | acc | finished-answer acc | cut |
|-----|-----|---------------------|-----|
| R0 clc | 0.6094 | 0.661 | 0.078 |
| R1 astra | 0.6133 | 0.730 | 0.160 |
| R3 clc_a4 | 0.6094 | 0.743 | 0.180 |
| R4 clc_a3 | 0.5781 | 0.787 | 0.266 |
| **R5 clc_sel** | **0.6641** | 0.702 | 0.055 |
| R6 mix | 0.5859 | 0.710 | 0.180 |

Highest overall is R5 (no tie), and R5 0.664 > R0 0.609, so it proceeds to confirmation. Matches `selected_fix.txt`. Caveat: the margin is 14 problems of 256 over R0 and 13 over R1, chosen as the max of five candidates.

## 3. Re-grading

Authors' `mathqa.extract` + `mathqa.is_correct` (math-verify) on every stored `text`, gold from data files.

- I2 (500): 0 disagreements in `pred`, 0 in `correct`; ids = math500 ids in order, no duplicates; acc 0.558.
- K2 (1319): 0 / 0; ids = gsm8k_test ids in order; acc 0.68992.
- Also re-graded the comparison arms B, I, A, E, L, K: 0 disagreements each.

## 4. Statistics (paired bootstrap, 10,000 resamples over problems, seed 0, 95% percentile CI, in percentage points)

| Contrast | mean | 95% CI | wins/losses | vs analysis_a4.json |
|----------|------|--------|-------------|--------------------|
| I2-B | +6.60 | [2.80, 10.40] | 64/31 | identical |
| I2-I | +2.80 | [-1.00, 6.60] | 55/41 | identical |
| I2-E | +10.00 | [5.40, 14.60] | 94/44 | identical |
| K2-L | +0.61 | [-1.82, 3.03] | 133/125 | identical |
| K2-K | +4.40 | [1.97, 6.82] | 162/104 | identical |

Max absolute difference to analysis_a4.json is 0.0 for mean and both CI bounds (sorted-id and dataset order give the same CI).

Gates: H11 (K2-L lower > -5) PASS (-1.82). H12 (K2-K > 0, lower > 0) PASS (1.97). H13 (I2-B lower > -5) PASS (2.80). Reported superiority: I2-B lower 2.80 > 0 PASS; K2-L lower -1.82 not > 0, so no superiority claim on GSM8K. All gate booleans agree with analysis_a4.json.

Arm stats: I2 0.558 (cut 0.172, 18,462 tok, finished-acc 0.671); B 0.492 / I 0.530 / E 0.458 / A 0.074. K2 0.690 (cut 0.0106, 9,336 tok, finished-acc 0.697); L 0.684 (finished-acc 0.723); K 0.646.

## 5. Leakage

reserve (256) and extract (512) vs math500 (500) and gsm8k_test (1319): id overlap 0, normalised exact problem-text overlap 0, 6-gram-shingle near-duplicates (containment >= 0.7) 0, in all four pairings. reserve vs extract id overlap 0. No duplicate ids inside any file. Sources: reserve 128 gsm8k_train + 128 math_train; extract 256 + 256, none test.

## 6. Edit and hardware

- `ckpt/clc_sel/model.safetensors` vs base release file: 320 tensors in ckpt, all present in base with identical dtype and shape; exactly two differ (torch.equal): `model.language_model.layers.15.mlp.down_proj.weight` and `...layers.19...`. Each is bf16 [1024, 3584] with exactly 64 changed columns (all 1024 rows). Delta is rank-1 (top singular values 1.707 / 1.408, next 0.0012 / 0.0009 = bf16 rounding). Relative change 0.1038 / 0.0816, equal to `edits.json`; bf16 sha256 matches `edits.json`; equals `edits/clc_sel.npz` cast to bf16.
- The checkpoint is a text-only repack: 168 base tensors are absent (the `model.visual.*` tower and `mtp.*` head). This is outside the edit, the same repackaging applies to the other ckpts (not verified here for the Astra ckpt), and no other retained tensor changed.
- edit args (edits/clc_sel.json): CLC directions, layers 15/19, alpha 6, k 64, select act, act_first = extract directions (loop-selective ranking), as the addendum describes.
- GPU: both summaries report `GPU-40333bf8-ac6a-447f-990a-b02c146800dc`, `NVIDIA A100-PCIE-40GB`; `nvidia-smi -L` on the host lists it as GPU 1 of two A100-PCIE-40GB (the in-container index 0 is host GPU 1). I2 and K2 ran concurrently on that same GPU (launched 4 min apart), consistent with launches.txt gpu=1 and the gpu_pre files. `cuda_visible` is null in both summaries, fine because `--gpus device=1` restricts the container.

## Observations (not failures)

- The H11-H13 gates pass, but the gain over CLC on MATH-500 (I2-I +2.8, CI includes 0) comes through fewer cut generations (17.2% vs 21.8%); finished-answer accuracy is 0.671 vs 0.673 for I and 0.723 for B. On GSM8K, finished-answer accuracy 0.697 is still below Astra's 0.723 and K2-L is not a superiority result. The addendum's stated motive (raise finished-answer accuracy) is not supported; the improvement is mostly fewer truncations.
- Selection used a 256-problem set with a 5-point lead; confirmation is on the same benchmarks used for CLC (as the addendum discloses).
- The first-reserve-run margin to the freeze file is under one second (still correct order).
