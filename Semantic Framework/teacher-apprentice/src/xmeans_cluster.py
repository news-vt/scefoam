#!/usr/bin/env python3
# File: src/xmeans_cluster.py

import sys
import json
import numpy as np
from sklearn.cluster import KMeans

def compute_bic(km: KMeans, X: np.ndarray) -> float:
    """
    Compute Bayesian Information Criterion for a fitted KMeans model.
    """
    centers = km.cluster_centers_
    labels  = km.labels_
    m, d    = X.shape
    K       = centers.shape[0]
    # spherical variance estimate
    var = np.sum((X - centers[labels])**2) / (m - K)
    if var <= 0:
        return -np.inf
    # log‐likelihood under spherical Gaussian
    ll = -0.5 * m * d * np.log(2 * np.pi * var) - 0.5 * (m - K) * d
    # BIC penalty term
    penalty = 0.5 * K * np.log(m) * (d + 1)
    return ll - penalty

def cluster(embeddings):
    """
    Sweep K from 2 up to min(10, n_samples), fit KMeans, and pick
    the one with the highest BIC. Return that model’s centers.
    """
    X = np.array(embeddings, dtype=float)
    n_samples, _ = X.shape

    # Too few points or degenerate → single center at the mean
    if n_samples < 2 or np.allclose(X, X[0]):
        return [X.mean(axis=0).tolist()]

    max_k = min(30, n_samples)
    best_bic = -np.inf
    best_centers = [X.mean(axis=0).tolist()]

    for K in range(2, max_k + 1):
        # Use n_init=10 to match scikit-learn’s default today
        km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
        bic = compute_bic(km, X)
        if bic > best_bic:
            best_bic = bic
            best_centers = km.cluster_centers_.tolist()

    return best_centers

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python xmeans_cluster.py <kb_path>"}))
        sys.exit(1)

    kb_path = sys.argv[1]
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        embeddings = kb.get('embeddings', [])
        centroids  = cluster(embeddings)
        print(json.dumps({"centroids": centroids}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
