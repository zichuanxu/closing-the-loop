#!/usr/bin/env bash
# Publish loopdir_001: research-state pointer and the portable CLC edit package.
set -euo pipefail
cd /home/xuzichuan/weightspace
cat >> docs/RESEARCH_STATE.md <<"EOF"

## Automated anti-loop edits vs. Weight-Space Programming (loopdir_001, 2026-11-04/05)

New direction after the handoff: a head-to-head with the blog Experiment 8 edit on Qwen3.5-0.8B (thinking, MATH-500).
Base 7.4 % and Astra 49.2 % reproduced (bit-exact edit rebuilt from the released diff). The pre-registered automated
method (recurrence direction, screened layers) failed (16.4 %). The diagnosed alternative, the closure-contrast edit
(loop state minus the model's own pre-`</think>` state, at Astra's layers/strength/sparsity), confirmed under a frozen
addendum: MATH-500 53.0 % (+45.6 vs base; +3.8 [-0.2, 7.8] vs Astra; +7.2 [2.8, 11.8] vs the default non-thinking
mode); GSM8K 64.6 % vs Astra 68.4 % (-3.8 [-6.1, -1.5], non-inferiority failed), with 35 % fewer tokens. Built in
~84 min of shared A100 time from 512 practice generations; no LLM agent, no gradients. No loop edit improved
non-thinking mode. Independent audit passed. Report: `docs/research/loopdir_results.md`; study: `experiments/loopdir_001/`;
package: `artifacts/exports/clc-loop-edit-v1.tar.gz`.
EOF
S=experiments/loopdir_001
E=artifacts/exports/clc-loop-edit-v1
mkdir -p "$E"
cp $S/edits/clcloc.npz $S/edits/clcloc.json $S/ckpt/clcloc/edits.json $S/code/build_ckpt.py $S/code/make_edit.py \
   $S/code/closure_dirs.py $S/code/extract_dirs.py $S/code/gen_eval.py $S/code/mcqa.py $S/RESULTS.md \
   $S/runs/frozen_clc.sha256 "$E/"
cat > "$E/README.md" <<"EOF"
# CLC loop edit for Qwen3.5-0.8B (thinking mode)

Two replacement tensors (`clcloc.npz`, float32; cast to bf16 on build): `model.language_model.layers.{15,19}.mlp.down_proj.weight`,
64 changed columns each. Build a text-only checkpoint from `Qwen/Qwen3.5-0.8B` @2fc06364715b967f1860aea9cf38778875588b17
plus the config / tokenizer / chat template in `RikiyaT/lm_understand-logs` exp8_math/base_model (mount paths in build_ckpt.py):

    python build_ckpt.py OUT --npz clcloc.npz

Rebuild the edit from scratch: gen_eval.py (512 base answers) -> extract_dirs.py -> closure_dirs.py ->
make_edit.py --dirs DIR/directions.npz --layers 15 19 --alpha 6 --k 64. Results, limits and protocol: RESULTS.md.
Use with thinking mode; it did not help non-thinking mode. Research artifact, not a general quality claim.
EOF
cd artifacts/exports
tar czf clc-loop-edit-v1.tar.gz clc-loop-edit-v1
sha256sum clc-loop-edit-v1.tar.gz | tee clc-loop-edit-v1.tar.gz.sha256
ls -la clc-loop-edit-v1.tar.gz
