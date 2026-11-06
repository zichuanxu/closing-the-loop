"""Decompose the GSM8K CLC-sel vs Astra gap by finish status (plain python3, seed 0)."""
import gzip, json, statistics as st, sys

def L(d):
    return {r["id"]: r for r in map(json.loads, gzip.open(f"runs/{d}/records.jsonl.gz", "rt"))}

A, C = L(sys.argv[1]), L(sys.argv[2])
ids = sorted(set(A) & set(C))
fin = lambda r: r["finish"] == "stop"
acc = lambda M, S: sum(M[i]["correct"] for i in S) / len(S)
med = lambda M, S: st.median(M[i]["n_tokens"] for i in S)
groups = {"both finished": lambda i: fin(A[i]) and fin(C[i]),
          "A cut, C finished": lambda i: not fin(A[i]) and fin(C[i]),
          "C cut, A finished": lambda i: fin(A[i]) and not fin(C[i]),
          "both cut": lambda i: not fin(A[i]) and not fin(C[i])}
for name, p in groups.items():
    S = [i for i in ids if p(i)]
    if S:
        print(f"{name}: n={len(S)} acc A={acc(A, S):.3f} C={acc(C, S):.3f} "
              f"median tok A={med(A, S):.0f} C={med(C, S):.0f}")
bf = [i for i in ids if fin(A[i]) and fin(C[i])]
aw = [i for i in bf if A[i]["correct"] and not C[i]["correct"]]
cw = [i for i in bf if C[i]["correct"] and not A[i]["correct"]]
print("both finished, discordant: A-only-right", len(aw), "C-only-right", len(cw))
for nm, S in (("A-only-right", aw), ("C-only-right", cw)):
    print(f"  {nm}: median tok A={med(A, S):.0f} C={med(C, S):.0f} "
          f"share C shorter={sum(C[i]['n_tokens'] < A[i]['n_tokens'] for i in S) / len(S):.2f}")
for nm, M in (("A", A), ("C", C)):
    r = [i for i in bf if M[i]["correct"]]; w = [i for i in bf if not M[i]["correct"]]
    print(f"{nm} both-finished median tok: right={med(M, r):.0f} wrong={med(M, w):.0f}")
