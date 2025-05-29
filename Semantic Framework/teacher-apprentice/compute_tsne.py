#!/usr/bin/env python3
"""
compute_tsne.py

Reads public/knowledge_base.json, runs a 2D t-SNE on
[token embeddings + centroid embeddings], and writes
public/tsne_data.json:

{
  "tokens": ["hello","world", …],
  "coords": [[x,y], …],
  "centroid_ids": ["1","2", …],
  "centroid_labels": ["animal","vehicle", …],
  "centroid_coords": [[x,y], …]
}

This version guards against:
 - Too few samples
 - All points identical
 - NaNs/infinite values
 - Any internal TSNE/PCA errors
"""
import json
import numpy as np
import warnings
import traceback
from pathlib import Path
from sklearn.manifold import TSNE
import sys

# Silence numeric warnings inside PCA/TSNE
warnings.filterwarnings("ignore")

# Paths
PORT = sys.argv[1] if len(sys.argv) > 1 else ""        # "" → fallback
SUFFIX = f"_{PORT}" if PORT else ""

BASE   = Path(__file__).parent
PUBLIC = BASE / "public"
KB     = PUBLIC / f"knowledge_base{SUFFIX}.json"
OUT    = PUBLIC / f"tsne_data{SUFFIX}.json"
# Load KB
with open(KB, "r", encoding="utf-8") as f:
    kb = json.load(f)

tokens          = kb.get("rawData", [])
embeddings      = np.array(kb.get("embeddings", []), dtype=float)
centroids       = np.array(list(kb.get("centroids", {}).values()), dtype=float)
cent_ids        = list(kb.get("centroids", {}).keys())
cent_labels_map = kb.get("centroidLabels", {})

# Make a parallel list of labels in the same order as cent_ids
centroid_labels = [cent_labels_map.get(cid, "") for cid in cent_ids]

# Combine embeddings + centroids
if embeddings.size or centroids.size:
    try:
        all_pts = np.vstack([embeddings, centroids])
    except Exception:
        all_pts = np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 100))
else:
    all_pts = np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 100))

n_samples = all_pts.shape[0]

# Prepare output
data = {
    "tokens":           tokens,
    "coords":           [],
    "centroid_ids":     cent_ids,
    "centroid_labels":  centroid_labels,
    "centroid_coords":  []
}

# Helper to decide if we should run TSNE
def can_run_tsne(pts):
    if pts.shape[0] < 2:
        return False
    if np.allclose(pts, pts[0], atol=1e-6):
        return False
    if not np.isfinite(pts).all():
        return False
    return True

if can_run_tsne(all_pts):
    perp = 3 if n_samples > 3 else max(1, n_samples - 1)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        Y    = tsne.fit_transform(all_pts)

        # Split embeddings vs centroids
        data["coords"]           = Y[: embeddings.shape[0]].tolist()
        data["centroid_coords"]  = Y[embeddings.shape[0] :].tolist()
    except Exception:
        print("Warning: TSNE computation failed:")
        traceback.print_exc()
else:
    print(f"compute_tsne.py: skipping TSNE (n_samples={n_samples})")

# Ensure output directory exists
OUT.parent.mkdir(parents=True, exist_ok=True)

# Write JSON
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
