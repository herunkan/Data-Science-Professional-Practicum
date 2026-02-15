from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def cluster_messages(
    df: pd.DataFrame,
    k_clusters: int = 6,
    max_features: int = 5000,
) -> Tuple[pd.DataFrame, Dict[int, List[str]], TfidfVectorizer, np.ndarray, KMeans]:
    if df.empty:
        raise ValueError("No posts available for clustering.")

    work = df.copy()
    corpus = work["clean_text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        X = vectorizer.fit_transform(corpus)
    if X.shape[0] < k_clusters:
        k_clusters = max(2, X.shape[0] // 2)
    if k_clusters < 2:
        k_clusters = 1

    model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    distances = pairwise_distances(X, model.cluster_centers_, metric="euclidean")
    row_idx = np.arange(X.shape[0])
    closest_dist = distances[row_idx, labels]

    work["cluster_id"] = labels
    work["centroid_distance"] = closest_dist

    # Cluster keywords from provided extracted keyword field.
    cluster_keywords: Dict[int, List[str]] = {}
    for cid in sorted(work["cluster_id"].unique()):
        kw_counter = Counter()
        subset = work[work["cluster_id"] == cid]
        for kw_csv in subset["keywords"].fillna(""):
            for kw in [k.strip() for k in kw_csv.split(",") if k.strip()]:
                kw_counter[kw] += 1
        cluster_keywords[int(cid)] = [w for w, _ in kw_counter.most_common(10)]

    return work, cluster_keywords, vectorizer, X.toarray(), model


def posts_near_centroid(df_clustered: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    return (
        df_clustered.sort_values(["cluster_id", "centroid_distance"], ascending=[True, True])
        .groupby("cluster_id")
        .head(top_n)
        .reset_index(drop=True)
    )


def save_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    if embeddings.shape[0] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(embeddings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    plt.title("Message Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
