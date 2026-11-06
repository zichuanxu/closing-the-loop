"""Build an edited Qwen3.5-0.8B checkpoint from an edit spec in ``edits/``.

    uv run --extra model python -m clc_sel.apply edits/clc_sel.json --out ckpt/clc_sel

Downloads ``Qwen/Qwen3.5-0.8B`` at the revision used in the paper, replaces the edited ``mlp.down_proj``
tensors and writes a complete model folder that vLLM and Transformers load as usual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from clc_sel.edit import apply_edit

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"


def load_spec(path: str | Path) -> dict[str, dict]:
    """``{tensor_name: {"u", "cols", "alpha"}}`` from a spec written by ``study/code/make_edit.py``."""
    spec = json.loads(Path(path).read_text())
    args, tensors = spec["args"], spec["tensors"]
    alphas = dict(zip(args["layers"], args["alphas"], strict=True)) if args.get("alphas") else {}
    out = {}
    for name, t in tensors.items():
        layer = int(name.split(".layers.")[1].split(".")[0])
        out[name] = {"u": t["u"], "cols": t["cols"], "alpha": alphas.get(layer, args["alpha"])}
    return out


def main() -> None:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="edit spec, e.g. edits/clc_sel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()

    src = Path(snapshot_download(a.model, revision=a.revision))
    out = Path(a.out)
    shutil.copytree(src, out, dirs_exist_ok=True)
    sd = load_file(src / WEIGHTS)
    report = {}
    for name, e in load_spec(a.spec).items():
        W = sd[name].float().numpy()
        edited = apply_edit(W, e["u"], e["cols"], e["alpha"]).astype(np.float32)
        sd[name] = torch.from_numpy(edited).to(torch.bfloat16).contiguous()
        report[name] = {
            "alpha": e["alpha"],
            "n_cols": len(e["cols"]),
            "bf16_sha256": hashlib.sha256(sd[name].view(torch.int16).numpy().tobytes()).hexdigest(),
        }
    (out / WEIGHTS).unlink()  # replace the hub symlink instead of writing through it
    save_file(sd, out / WEIGHTS, metadata={"format": "pt"})
    (out / "clc_sel_edit.json").write_text(json.dumps({"spec": str(a.spec), "tensors": report}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
