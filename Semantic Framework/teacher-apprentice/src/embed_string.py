#!/usr/bin/env python3
"""
embed_string.py

Takes an input string via stdin or argument, tokenizes it,
and returns a JSON list of embeddings per token using a pre-cached
GloVe (glove-wiki-gigaword-100) model.
"""

import sys, json, re
import numpy as np
from gensim.downloader import load  # <-- import here

# Load once at import, so we never hit the downloader at runtime
model = load('glove-wiki-gigaword-100')


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9_]", " ", text)
    return [tok for tok in text.split() if tok]


def embed_tokens(tokens):
    size = model.vector_size
    out = []
    for tok in tokens:
        if tok in model.key_to_index:
            out.append(model[tok].tolist())
        else:
            out.append([0.0] * size)
    return out


def main():
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = sys.stdin.read().strip()

    tokens     = tokenize(text)
    embeddings = embed_tokens(tokens)

    print(json.dumps({
        "tokens":     tokens,
        "embeddings": embeddings
    }))


if __name__ == "__main__":
    main()
