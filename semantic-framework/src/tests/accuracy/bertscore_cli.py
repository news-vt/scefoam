#!/usr/bin/env python3
# pip install bert-score torch transformers

import sys

# lazy import & auto‑install
try:
    from bert_score import score as bert_score
except ImportError:
    import subprocess, sys as _s
    subprocess.check_call(
        [_s.executable, "-m", "pip", "install", "bert-score", "torch", "transformers"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    from bert_score import score as bert_score

if len(sys.argv) != 3:
    sys.exit("usage: bertscore_cli.py <ref.txt> <hyp.txt>")

# load the two files
with open(sys.argv[1], encoding='utf8') as f:
    refs = [f.read().strip()]
with open(sys.argv[2], encoding='utf8') as f:
    hyps = [f.read().strip()]

# compute P/R/F (we only care about F1)
P, R, F = bert_score(hyps, refs, lang='en', verbose=False)

# emit just the float on stdout
print(F[0].item())
