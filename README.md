# Closing the Loop

**A simple rule beats an agent-written weight edit for reasoning loops.**

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen3.5--0.8B--CLC--sel-yellow)](https://huggingface.co/zichuanxu/Qwen3.5-0.8B-CLC-sel)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-all%20generations-yellow)](https://huggingface.co/datasets/zichuanxu/closing-the-loop-generations)
[![CI](https://github.com/zichuanxu/closing-the-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/zichuanxu/closing-the-loop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Small reasoning models often fail by repeating the same text until they run out of tokens. In thinking mode,
Qwen3.5-0.8B truncates 93% of its MATH-500 answers at 32,768 tokens and solves 7.4%.
[Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/) let an LLM agent read the model's
weights for about 20 hours, and the agent wrote a rank-one edit (the "Astra edit") that stops many of these loops.

**CLC-sel** (loop-selective closure contrast) is a fixed rule that builds an edit of the same form and size with no
agent, gradients or correctness labels. It runs in about 84 GPU-minutes on 512 of the model's own answers to practice
problems, and it scores **56.3% on MATH-500**, 6.7 points above the agent-written edit.

This repository has the edit rule, the released edits and the full experiment record. The edited model and every
generation are on Hugging Face:

| | Hugging Face | Contents |
|---|---|---|
| Model | [zichuanxu/Qwen3.5-0.8B-CLC-sel](https://huggingface.co/zichuanxu/Qwen3.5-0.8B-CLC-sel) | The exact checkpoint evaluated (bf16, text only, Apache 2.0) |
| Dataset | [zichuanxu/closing-the-loop-generations](https://huggingface.co/datasets/zichuanxu/closing-the-loop-generations) | Full texts, grades and summaries for all 69 runs (175 MB) |

## Results

Qwen3.5-0.8B in thinking mode. Both edits change layers 15 and 19 with α = 6 and 64 neurons per layer.
Rows average three seeds; Δ is the paired difference with a 95% bootstrap interval over problems.

| Method | MATH-500 acc. | Truncated | Δ vs. Astra | GSM8K acc. | Truncated | Δ vs. Astra |
|---|---:|---:|---|---:|---:|---|
| Base model (seed 0) | 7.4 | 93.0 | | 13.8¹ | | |
| Astra edit | 49.7 | 32.1 | | 67.4 | 6.9 | |
| **CLC-sel (ours)** | **56.3** | **17.7** | **+6.7** [4.5, 8.9] | **69.0** | **1.0** | **+1.6** [0.2, 3.0] |

¹ Reported by Takehi et al.

- On GSM8K, CLC-sel also uses 24% fewer tokens than the Astra edit.
- CLC-sel beats budget forcing on the unedited model by 20.8 points on MATH-500, and its lead holds when both edits
  use budget forcing.
- With layers and strength chosen by our own rules (a generation screen and a closed-form strength), the edit
  matches the Astra edit on both benchmarks.
- The gain comes from answers that finish instead of looping. On problems the model already finishes, the edit does
  not make it reason better.

Every number can be checked against the run summaries in [`study/runs/`](study/runs) and the generations on
Hugging Face.

## Method

CLC-sel edits the MLP down projection `W` of a layer with one rank-one update:

```
W' = W − α u (m ⊙ s)ᵀ,   s = Wᵀu
```

- **Direction `u`.** The unit vector from the mean hidden state in the 64 tokens before the model's own `</think>`
  to the mean state inside its own verbatim loops. With α > 1 the edit pushes loop states toward the state in which
  the model concludes.
- **Neurons `m`.** The 64 neurons whose write along `u` rises most when the same text repeats, ranked by
  `(ā_rep − ā_first) · s`. The edit thus acts mainly inside loops and ends fewer normal answers early.
- **Strength α.** Taken from the Astra edit (α = 6). A closed-form rule, `α* = gap / Σ_c s_c (ā_rep,c − ā_pre,c)`,
  clipped to [0.5, 16], replaces it in the fully automatic variant.

## Quick start

### Use the edited model

```python
from vllm import LLM, SamplingParams

llm = LLM("zichuanxu/Qwen3.5-0.8B-CLC-sel")
params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768)
problem = "What is the smallest positive integer n such that 3n is a perfect square and 2n is a perfect cube?"
prompt = problem + " Please reason step by step, and put your final answer within \\boxed{}."
out = llm.chat([{"role": "user", "content": prompt}], params, chat_template_kwargs={"enable_thinking": True})
print(out[0].outputs[0].text)
```

These are the prompt and sampling settings of all thinking-mode runs. The edit was evaluated in thinking mode only.

### Inspect the generations

```python
import gzip, json
from huggingface_hub import hf_hub_download

path = hf_hub_download("zichuanxu/closing-the-loop-generations", "m500_I2_clcsel/records.jsonl.gz",
                       repo_type="dataset")
rows = [json.loads(line) for line in gzip.open(path, "rt")]
print(sum(r["correct"] for r in rows) / len(rows))  # 0.558, CLC-sel on MATH-500, seed 0
```

Each record has `id`, `correct`, `pred`, `finish` (`stop` or `length`), `n_tokens` and the full `text`. The
[dataset card](https://huggingface.co/datasets/zichuanxu/closing-the-loop-generations) maps run names to methods.

### Install and rebuild the edit

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zichuanxu/closing-the-loop && cd closing-the-loop
uv sync
uv run pytest
```

Rebuild an edited model from a spec in [`edits/`](edits). This downloads `Qwen/Qwen3.5-0.8B` at revision `2fc06364`
and writes the same edited tensors, bit for bit:

```bash
uv run --extra model python -m clc_sel.apply edits/clc_sel.json --out ckpt/clc_sel
```

The edit math is a few NumPy functions in [`src/clc_sel/edit.py`](src/clc_sel/edit.py):

```python
from clc_sel import apply_edit, closed_form_alpha, closure_direction, select_neurons

u = closure_direction(h_loop, h_close)  # residual-stream window means of one layer
cols = select_neurons(W, u, a_rep, a_first, k=64)  # neuron activations on repeats and first occurrences
W_edited = apply_edit(W, u, cols, alpha=6.0)
```

## Released edits

Each spec stores, per edited tensor, the direction `u` (1,024 floats), the 64 neuron indices and the strength. The
base weights plus the spec define the edit exactly. Only CLC-sel is on Hugging Face as a full checkpoint; build the
others with `clc_sel.apply`.

| File | Edit | Layers | α |
|---|---|---|---|
| `clc_sel.json` | CLC-sel, the main method | 15, 19 | 6 |
| `clc.json` | CLC, neurons ranked by their write on loops only | 15, 19 | 6 |
| `auto_layer13.json` | CLC-sel with automatic layer and strength | 13 | 11.82 |

In `clc.json` the `u_from` field reads `astra`, but no Astra checkpoint was given, so `u` comes from `--dirs` as for
the other specs.

## Repository layout

```
src/clc_sel/     edit rule (NumPy) and checkpoint builder
tests/           unit tests
edits/           released edit specs
study/           the experiment record, kept as it was run
  protocol/      protocol and addenda, frozen before each confirmatory run, with their hashes
  code/          every script that produced a result
  docker/        container images (vLLM 0.30.0, Math-Verify 0.9.0)
  data/          problem sets with SHA-256 sums
  runs/          per-run summaries (accuracy, truncation, tokens, GPU, seed) and analysis outputs
  audits/        two independent re-grading and statistics audits
  RESULTS.md     running lab notebook
```

## Reproducing the study

All runs used NVIDIA A100 40 GB GPUs. The scripts in [`study/code/`](study/code) are the exact versions that
produced the reported numbers. They assume the container mounts in [`study/code/run_gpu.sh`](study/code/run_gpu.sh):
the study folder at `/work`, the model at `/models/Qwen3.5-0.8B` and the Astra logs
([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) @4d09ba6d) at `/authors`, which
supply the grader and the text-only model config.

1. `gen_eval.py`: 512 base-model answers to the `extract` practice problems.
2. `extract_dirs.py`: loop windows and neuron activations. `closure_dirs.py`: pre-closure states and the direction.
3. `make_edit.py --dirs runs/clc/directions.npz --u-from dirs --act-first runs/extract/directions.npz --layers 15 19 --alpha 6 --k 64 --out edits/clc_sel.npz`
4. `build_ckpt.py`, then `gen_eval.py` on `math500.jsonl` and `gsm8k_test.jsonl`. `analyze.py` and `boot10k.py`
   compute the paired bootstrap.

The `chain*.sh` files are the launch sequences in the order they ran. The generations and the edited checkpoint are
too large for git and are on Hugging Face.

## Citation

```bibtex
@misc{xu2026closingtheloop,
  title  = {Closing the Loop: A Simple Rule Beats an Agent-Written Weight Edit for Reasoning Loops},
  author = {Xu, Zichuan},
  year   = {2026},
  url    = {https://github.com/zichuanxu/closing-the-loop}
}
```

## Acknowledgements and license

The Astra edit, its logs and the evaluation protocol are by
[Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/). Code is [MIT](LICENSE). The edited
model follows the Apache 2.0 license of Qwen3.5-0.8B. The problem sets come from MATH and GSM8K under their own
licenses.
