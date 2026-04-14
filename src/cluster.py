"""Stage 3: PCA + KMeans clustering with auto-k via silhouette + MMR representative sampling."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.embed import cosine_similarity_matrix

PCA_DIM = 50
K_MIN = 6
K_MAX = 10
SILHOUETTE_SAMPLE_SIZE = 1500
MIN_CLUSTER_SIZE = 20  # smaller clusters merge into "기타"
OTHER_CLUSTER_ID = -1
REPRESENTATIVES_PER_CLUSTER = 10  # 5 nearest + 5 MMR diverse
MMR_LAMBDA = 0.6


@dataclass
class ClusterInfo:
    id: int
    size: int
    representatives: list  # review_ids


@dataclass
class ClusterStats:
    k: int
    silhouette: float
    cluster_sizes: dict
    other_count: int
    pca_dim: int


def _select_k(reduced: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> tuple[int, float, dict]:
    """Search k_min..k_max, pick k with highest silhouette score on a sample."""
    n = len(reduced)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=sample_size, replace=False)
    sample = reduced[idx]

    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(sample)
        if len(set(labels)) < 2:
            scores[k] = -1.0
            continue
        score = silhouette_score(sample, labels, metric="euclidean")
        scores[k] = float(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score, scores


def _mmr_select(
    cluster_emb: np.ndarray, exclude: set[int], n_select: int, lambda_: float = MMR_LAMBDA
) -> list[int]:
    """Select n_select diverse indices via Maximal Marginal Relevance.

    Relevance = similarity to centroid; diversity = inverse max similarity to selected set.
    Returns indices into cluster_emb (NOT the global index).
    """
    n = len(cluster_emb)
    if n == 0:
        return []
    centroid = cluster_emb.mean(axis=0, keepdims=True)
    sim_to_centroid = cosine_similarity_matrix(cluster_emb, centroid).flatten()
    sim_pairwise = cosine_similarity_matrix(cluster_emb, cluster_emb)

    selected: list[int] = []
    candidates = [i for i in range(n) if i not in exclude]
    while len(selected) < n_select and candidates:
        best_i = candidates[0]
        best_score = -1e9
        for i in candidates:
            relevance = sim_to_centroid[i]
            diversity_penalty = (
                max(sim_pairwise[i, j] for j in selected) if selected else 0.0
            )
            score = lambda_ * relevance - (1 - lambda_) * diversity_penalty
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def _pick_representatives(
    cluster_member_global_idx: np.ndarray,
    cluster_emb: np.ndarray,
    review_ids: np.ndarray,
    n_total: int = REPRESENTATIVES_PER_CLUSTER,
) -> list:
    n_nearest = n_total // 2
    n_mmr = n_total - n_nearest
    centroid = cluster_emb.mean(axis=0, keepdims=True)
    sim = cosine_similarity_matrix(cluster_emb, centroid).flatten()
    nearest_local = np.argsort(-sim)[:n_nearest].tolist()
    mmr_local = _mmr_select(cluster_emb, exclude=set(nearest_local), n_select=n_mmr)
    chosen_local = nearest_local + mmr_local
    chosen_ids = [int(review_ids[cluster_member_global_idx[li]]) for li in chosen_local]
    return chosen_ids


def run_clustering(
    embeddings: np.ndarray,
    review_ids: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> tuple[np.ndarray, list[ClusterInfo], ClusterStats]:
    """Run PCA + KMeans + size-based merging + representative sampling.

    Returns (assignments aligned with embeddings rows, list of ClusterInfo, stats).
    Assignments use OTHER_CLUSTER_ID (-1) for tiny merged clusters.
    """
    n = len(embeddings)
    reduced = PCA(n_components=min(PCA_DIM, n - 1, embeddings.shape[1]), random_state=42).fit_transform(embeddings)

    best_k, best_score, _all_scores = _select_k(reduced, k_min=k_min, k_max=k_max)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    raw_labels = km.fit_predict(reduced)

    assignments = raw_labels.copy()
    cluster_sizes = pd.Series(raw_labels).value_counts().to_dict()
    other_count = 0
    for cid, size in cluster_sizes.items():
        if size < MIN_CLUSTER_SIZE:
            assignments[assignments == cid] = OTHER_CLUSTER_ID
            other_count += size

    surviving_ids = sorted(set(int(c) for c in assignments) - {OTHER_CLUSTER_ID})

    clusters: list[ClusterInfo] = []
    for cid in surviving_ids:
        member_mask = assignments == cid
        member_global_idx = np.where(member_mask)[0]
        cluster_emb = embeddings[member_mask]
        reps = _pick_representatives(member_global_idx, cluster_emb, review_ids)
        clusters.append(ClusterInfo(id=int(cid), size=int(member_mask.sum()), representatives=reps))

    if other_count > 0:
        member_mask = assignments == OTHER_CLUSTER_ID
        member_global_idx = np.where(member_mask)[0]
        cluster_emb = embeddings[member_mask]
        reps = _pick_representatives(member_global_idx, cluster_emb, review_ids)
        clusters.append(ClusterInfo(id=OTHER_CLUSTER_ID, size=int(member_mask.sum()), representatives=reps))

    final_sizes = {int(c.id): int(c.size) for c in clusters}
    stats = ClusterStats(
        k=int(best_k),
        silhouette=float(best_score),
        cluster_sizes=final_sizes,
        other_count=int(other_count),
        pca_dim=int(reduced.shape[1]),
    )
    return assignments, clusters, stats


def save_cluster_artifact(
    assignments: np.ndarray,
    clusters: list[ClusterInfo],
    stats: ClusterStats,
    review_ids: np.ndarray,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "03_clusters.json"
    payload = {
        "stats": asdict(stats),
        "clusters": [asdict(c) for c in clusters],
        "assignments": {int(rid): int(cid) for rid, cid in zip(review_ids, assignments)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_cluster_artifact(out_dir: Path) -> tuple[dict, list[ClusterInfo], ClusterStats]:
    payload = json.loads((out_dir / "03_clusters.json").read_text(encoding="utf-8"))
    clusters = [ClusterInfo(**c) for c in payload["clusters"]]
    stats = ClusterStats(**payload["stats"])
    assignments = {int(k): int(v) for k, v in payload["assignments"].items()}
    return assignments, clusters, stats
