# Closing the Loop

Code, edits, data and the full experiment record for
**Closing the Loop: A Simple Rule Beats an Agent-Written Weight Edit for Reasoning Loops**.

Small reasoning models often fail by repeating the same text until they run out of tokens. In thinking mode,
Qwen3.5-0.8B truncates 93% of its MATH-500 answers at 32,768 tokens. [Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/)
let an LLM agent read the model's weights for about 20 hours, and the agent wrote a rank-one edit (the "Astra edit")
that stops many of these loops. We show that a fixed rule, with no agent, gradients or correctness labels,
builds a better edit of the same form and size.

**Loop-selective closure contrast (CLC-sel)** edits an MLP down projection `W` of one layer:

```
W' = W - α u (m ⊙ s)ᵀ,   s = Wᵀu
```

- `u` is the unit direction from the mean hidden state in the 64 tokens before the model's own `</think>`
  to the mean state inside its own verbatim loops.
- `m` keeps the 64 neurons whose write along `u` rises most when the same text repeats, ranked by `(ā_rep − ā_first) · s`.
- The rule runs in about 84 GPU-minutes on 512 of the model's own answers to practice problems.

## Results

The paper is forthcoming; every number below is backed by the run summaries in `study/runs/`.

Qwen3.5-0.8B in thinking mode, layers 15 and 19, α = 6, 64 neurons per layer (the Astra edit's settings).
Three seeds; Δ is the paired difference with a 95% bootstrap interval over problems.

| Method | MATH-500 acc. | Truncated | Δ vs. Astra | GSM8K acc. | Truncated | Δ vs. Astra |
|---|---:|---:|---|---:|---:|---|
| Base model (seed 0) | 7.4 | 93.0 | | 13.8¹ | | |
| Astra edit | 49.7 | 32.1 | | 67.4 | 6.9 | |
| **CLC-sel (ours)** | **56.3** | **17.7** | **+6.7** [4.5, 8.9] | **69.0** | **1.0** | **+1.6** [0.2, 3.0] |

¹ Reported by Takehi et al.

- CLC-sel beats budget forcing on the unedited model by 20.8 points on MATH-500, and its lead holds when both edits
  use budget forcing.
- With layers and strength chosen by our own rules (a generation screen and a closed-form strength), the edit
  matches the Astra edit on both benchmarks.
- The gain comes from answers that finish instead of looping. On problems the model already finishes, the edit does
  not make it reason better.

## Hugging Face

- Edited model: [zichuanxu/Qwen3.5-0.8B-CLC-sel](https://huggingface.co/zichuanxu/Qwen3.5-0.8B-CLC-sel), the exact checkpoint evaluated.
- Every generation: [zichuanxu/closing-the-loop-generations](https://huggingface.co/datasets/zichuanxu/closing-the-loop-generations), full texts and grades for all 69 runs.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zichuanxu/closing-the-loop && cd closing-the-loop
uv sync
uv run pytest
```

Build the edited model from the released edit spec. This downloads `Qwen/Qwen3.5-0.8B` at revision `2fc06364`:

```bash
uv run --extra model python -m clc_sel.apply edits/clc_sel.json --out ckpt/clc_sel
```

The output folder loads like the original model in vLLM or Transformers. Use thinking mode with the sampling
settings used in all experiments: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 tokens.

The edit math is a few NumPy functions in [`src/clc_sel/edit.py`](src/clc_sel/edit.py):

```python
from clc_sel import closure_direction, select_neurons, closed_form_alpha, apply_edit

u = closure_direction(h_loop, h_close)  # residual-stream window means of one layer
cols = select_neurons(W, u, a_rep, a_first, k=64)  # neuron activations on repeats and first occurrences
W_edited = apply_edit(W, u, cols, alpha=6.0)
```

## Released edits

Each spec in [`edits/`](edits) stores, per edited tensor, the direction `u` (1,024 floats), the 64 neuron indices and
the strength. The base weights plus the spec define the edit exactly. In `clc.json` the `u_from` field reads
`astra`, but no Astra checkpoint was given, so `u` comes from `--dirs` as for the other specs.

| File | Edit | Layers | α |
|---|---|---|---|
| `clc_sel.json` | CLC-sel, the main method | 15, 19 | 6 |
| `clc.json` | CLC, neurons ranked by their write on loops only | 15, 19 | 6 |
| `auto_layer13.json` | CLC-sel with automatic layer and strength | 13 | 11.82 |

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
produced the reported numbers and assume the container mounts set up in
[`study/code/run_gpu.sh`](study/code/run_gpu.sh): the study folder at `/work`, the model at `/models/Qwen3.5-0.8B`
and the authors' logs ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) @4d09ba6d) at
`/authors`, which supplies the grader and the text-only model config. The pipeline is:

1. `gen_eval.py`: 512 base-model answers to the `extract` practice problems.
2. `extract_dirs.py`: loop windows and neuron activations; `closure_dirs.py`: pre-closure states and the direction.
3. `make_edit.py --dirs runs/clc/directions.npz --u-from dirs --act-first runs/extract/directions.npz --layers 15 19 --alpha 6 --k 64 --out edits/clc_sel.npz`
4. `build_ckpt.py`, then `gen_eval.py` on `math500.jsonl` and `gsm8k_test.jsonl`; `analyze.py` and `boot10k.py` for the paired bootstrap.

The `chain*.sh` files are the launch sequences in the order they ran. Full generations (about 200 MB) and the edited
checkpoints are not in git; they are on Hugging Face (above).

## Citation

```bibtex
@misc{xu2026closingtheloop,
  title  = {Closing the Loop: A Simple Rule Beats an Agent-Written Weight Edit for Reasoning Loops},
  author = {Xu, Zichuan},
  year   = {2026},
  url    = {https://github.com/zichuanxu/closing-the-loop}
}
```

## License

Code: [MIT](LICENSE). The problem sets come from MATH and GSM8K under their own licenses. The Astra edit and the
evaluation protocol are by Takehi et al.
