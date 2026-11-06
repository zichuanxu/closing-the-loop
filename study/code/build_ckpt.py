"""Text-only Qwen3.5-0.8B checkpoint, optionally with edited tensors.

  python build_ckpt.py OUT [--astra EDIT_DIR] [--npz EDITS.npz]

Base: the release file of Qwen/Qwen3.5-0.8B @2fc06364 (only model.language_model.* kept) plus the authors' text-only
config / tokenizer config / chat template (lm_understand-logs exp8_math/base_model). --astra rebuilds an Astra edit with
the authors' hash-checked apply_edit.py. --npz takes {tensor_name: float32 array} of full replacement tensors (our edits).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

MODEL = "/models/Qwen3.5-0.8B"
RELEASE = f"{MODEL}/model.safetensors-00001-of-00001.safetensors"
AUTH = "/authors/exp8_math"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--astra")
    ap.add_argument("--npz")
    a = ap.parse_args()
    sd = {k: v for k, v in load_file(RELEASE).items() if k.startswith("model.language_model.")}
    edits = {}
    if a.astra:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sys.executable, f"{AUTH}/code/apply_edit.py", RELEASE, a.astra, td], check=True)
            edits.update({f[:-4]: np.load(os.path.join(td, f)) for f in os.listdir(td) if f.endswith(".npy")})
    if a.npz:
        z = np.load(a.npz)
        edits.update({k: z[k] for k in z.files})
    rel = {}
    for name, arr in edits.items():
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
        assert t.shape == sd[name].shape, (name, t.shape, sd[name].shape)
        rel[name] = float((t.float() - sd[name].float()).norm() / sd[name].float().norm())
        sd[name] = t.contiguous()
    os.makedirs(a.out, exist_ok=True)
    save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(f"{AUTH}/base_model/Qwen3.5-0.8B-text/{f}", os.path.join(a.out, f))
    for f in ("tokenizer.json", "vocab.json", "merges.txt"):
        shutil.copyfile(f"{MODEL}/{f}", os.path.join(a.out, f))
    hashes = {n: hashlib.sha256(sd[n].view(torch.int16).numpy().tobytes()).hexdigest() for n in edits}
    json.dump({"astra": a.astra, "npz": a.npz, "relative_change": rel, "edited_bf16_sha256": hashes},
              open(os.path.join(a.out, "edits.json"), "w"), indent=1)
    print(f"{len(sd)} tensors, {len(edits)} edited -> {a.out}", rel)


if __name__ == "__main__":
    main()
