"""Generate + grade math answers with vLLM, the authors' exp8 protocol (think35 / nothink35), graded by their mathqa.

  python gen_eval.py --ckpt DIR --data X.jsonl --out DIR [--n 1] [--limit K] [--protocol think35] [--max-tokens 32768]

data rows: {"id", "problem", "answer"}. Writes records.jsonl.gz (every sample, full text), summary.json.
"""
import argparse, gzip, json, os, subprocess, sys, time

sys.path.insert(0, "/authors/exp8_math/code")
import mathqa as Q  # noqa: E402  (authors' grader: last \\boxed{}, math-verify 0.9.0)

PROTOCOLS = {  # authors' code/evaluate.py
    "think35": ({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, True),
    "nothink35": ({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}, False),
}


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--protocol", default="think35")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem", type=float, default=0.5)
    ap.add_argument("--rep-penalty", type=float, help="vLLM repetition_penalty (addendum 6)")
    ap.add_argument("--force-close", type=int, help="s1 budget forcing: thinking cap; rest of --max-tokens answers")
    ap.add_argument("--nowait", action="store_true", help="addendum 9: NoWait keyword suppression (logit bias -100)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data)][a.offset:]
    rows = rows[:a.limit] if a.limit else rows
    samp, thinking = PROTOCOLS[a.protocol]
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    t0 = time.time()
    llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=a.mem, max_model_len=a.max_tokens + 2048,
              seed=a.seed)
    gpu_loaded = gpu_info()
    prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking) for r in rows]
    if a.rep_penalty:  # addendum 6: CTRL repetition penalty on top of the protocol
        samp = {**samp, "repetition_penalty": a.rep_penalty}
    banned = []
    if a.nowait:  # addendum 9: NoWait (Wang et al. 2025), exact word match after stripping non [a-z-] characters
        import re
        kw = {"wait", "alternatively", "hmm", "but", "however", "alternative", "another", "check", "double-check", "oh",
              "maybe", "verify", "other", "again", "now", "ah", "any"}
        banned = [t for t in range(len(tok)) if re.sub(r"[^a-z-]", "", tok.decode([t]).lower()) in kw]
        samp = {**samp, "logit_bias": {t: -100.0 for t in banned}}
        print("nowait banned", len(banned), sorted({tok.decode([t]) for t in banned})[:80], flush=True)
    budget = a.force_close or a.max_tokens
    sp = SamplingParams(n=a.n, max_tokens=budget, seed=a.seed, **samp)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    forced = {}
    if a.force_close:  # addendum 6: s1 budget forcing, append </think> at the thinking cap and let it answer
        from vllm.inputs import TokensPrompt
        assert a.n == 1
        suffix = tok("\n</think>\n\n", add_special_tokens=False)["input_ids"]
        idx = [i for i, o in enumerate(outs) if o.outputs[0].finish_reason == "length"]
        sfx = {i for i in idx if "</think>" not in outs[i].outputs[0].text}  # already-closed ones just continue
        sp2 = SamplingParams(n=1, max_tokens=a.max_tokens - a.force_close, seed=a.seed, **samp)
        outs2 = llm.generate([TokensPrompt(prompt_token_ids=list(outs[i].prompt_token_ids) + list(outs[i].outputs[0].token_ids)
                                           + (suffix if i in sfx else [])) for i in idx], sp2) if idx else []
        forced = {i: (o2.outputs[0], i in sfx) for i, o2 in zip(idx, outs2)}
    t2 = time.time()
    gpu_end = gpu_info()
    ok = cut = toks = tot = 0
    banned_set, banned_seen = set(banned), 0
    with gzip.open(os.path.join(a.out, "records.jsonl.gz"), "wt") as f:
        for i, (r, o) in enumerate(zip(rows, outs)):
            for k, s in enumerate(o.outputs):
                assert s.finish_reason in ("stop", "length"), s.finish_reason
                text, n_tok, finish = s.text, len(s.token_ids), s.finish_reason
                if i in forced:
                    s2, added = forced[i]
                    text = text + ("\n</think>\n\n" if added else "") + s2.text
                    n_tok, finish = n_tok + len(s2.token_ids), s2.finish_reason
                n_banned = sum(t in banned_set for t in s.token_ids) if banned else 0
                pred, _ = Q.extract(text)
                good = bool(Q.is_correct(pred, r["answer"]))
                ok += good; cut += finish == "length"; toks += n_tok; tot += 1; banned_seen += n_banned
                f.write(json.dumps({"id": r["id"], "sample": k, "correct": good, "pred": pred, "finish": finish,
                                    "n_tokens": n_tok, "closed_think": "</think>" in text, "forced": i in forced and forced[i][1],
                                    "text": text}) + "\n")
    summ = {"ckpt": a.ckpt, "data": a.data, "offset": a.offset, "problems": len(rows), "n": a.n, "protocol": a.protocol,
            "sampling": samp, "thinking": thinking, "max_tokens": a.max_tokens, "seed": a.seed, "acc": ok / tot,
            "force_close": a.force_close, "nowait_banned_ids": len(banned), "nowait_banned_seen": banned_seen, "forced_share": sum(v[1] for v in forced.values()) / tot,
            "cut_share": cut / tot, "mean_tokens": toks / tot, "tokens_sum": toks, "gen_seconds": round(t2 - t1, 1),
            "load_seconds": round(t1 - t0, 1), "tok_per_s": round(toks / (t2 - t1), 1),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_after_load": gpu_loaded, "gpu_end": gpu_end}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
