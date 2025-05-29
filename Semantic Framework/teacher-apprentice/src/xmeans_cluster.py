#!/usr/bin/env python3
# File: src/xmeans_cluster.py
#
# Stand-alone X-means (Pelleg & Moore, 2000) with spherical Gaussian BIC.

import sys, json, numpy as np
from pathlib import Path
from typing   import List, Tuple
from sklearn.cluster import KMeans


# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def spherical_bic(x: np.ndarray,
                  labels: np.ndarray,
                  centers: np.ndarray) -> float:
    """BIC of a spherical-covariance mixture."""
    n, d   = x.shape
    k      = centers.shape[0]
    var    = np.sum((x - centers[labels]) ** 2) / (n - k)
    if var <= 0 or not np.isfinite(var):
        return -np.inf
    log_lh = -0.5 * n * d * np.log(2 * np.pi * var) \
             -0.5 * (n - k) * d
    n_params = k * (d + 1)        # mean (d) + variance (1) per cluster
    return log_lh - 0.5 * n_params * np.log(n)


def kmeans2(x: np.ndarray,
            init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run KMeans with fixed K=2 and given 2×d initialise centers."""
    km = KMeans(n_clusters=2, init=init, n_init=1, max_iter=100,
                algorithm="lloyd", random_state=42).fit(x)
    return km.labels_, km.cluster_centers_


# ────────────────────────────────────────────────────────────────
# X-means core
# ────────────────────────────────────────────────────────────────
def xmeans(x: np.ndarray,
           k_init: int = 1,
           k_max:  int = 30) -> List[np.ndarray]:
    """Return a list of final cluster centres."""
    n, _ = x.shape
    if n <= 2:
        return [x.mean(0)]

    # start with vanilla KMeans(k_init)
    km  = KMeans(n_clusters=k_init, random_state=42, n_init="auto").fit(x)
    clusters = [(x[km.labels_ == i], km.cluster_centers_[i])
                for i in range(k_init)]

    improved = True
    while improved and len(clusters) < k_max:
        improved = False
        new_clusters: List[Tuple[np.ndarray, np.ndarray]] = []

        for pts, ctr in clusters:
            # skip tiny clusters
            if pts.shape[0] <= 3:
                new_clusters.append((pts, ctr))
                continue

            # propose a 2-way split along the PCA-major axis
            v = np.linalg.svd(pts - ctr, full_matrices=False)[2][0]
            init = np.stack([ctr + v, ctr - v])
            labels2, centers2 = kmeans2(pts, init)

            bic_parent = spherical_bic(pts, np.zeros(len(pts), int),
                                       ctr[None, :])
            bic_split  = spherical_bic(pts, labels2, centers2)

            if bic_split > bic_parent:            # accept split
                new_clusters.extend([
                    (pts[labels2 == 0], centers2[0]),
                    (pts[labels2 == 1], centers2[1])
                ])
                improved = True
            else:                                 # keep as-is
                new_clusters.append((pts, ctr))

        clusters = new_clusters

    return [ctr.tolist() for _, ctr in clusters]


# ────────────────────────────────────────────────────────────────
# CLI wrapper
# ────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python xmeans_cluster.py <kb_path>"}))
        sys.exit(1)

    kb_path = Path(sys.argv[1])
    try:
        kb   = json.loads(kb_path.read_text(encoding="utf-8"))
        emb  = np.asarray(kb.get("embeddings", []), dtype=float)
        if emb.ndim != 2 or emb.shape[0] == 0:
            raise ValueError("No embeddings in KB")

        centers = xmeans(emb)
        print(json.dumps({"centroids": centers}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
