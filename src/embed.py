"""Stage 2a: Batch embed all review texts via OpenAI text-embedding-3-small."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.llm import CostTracker, get_client

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 256  # well under OpenAI's 2048 input limit per request


def embed_texts(
    texts: list[str],
    cost: CostTracker,
    step: str = "embedding",
    progress_cb=None,
) -> np.ndarray:
    """Embed in batches. Returns (n, EMBED_DIM) float32 array.

    progress_cb(done: int, total: int) called after each batch if provided.
    """
    client = get_client()
    n = len(texts)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)

    cleaned = [t if t and t.strip() else " " for t in texts]

    done = 0
    for start in range(0, n, BATCH_SIZE):
        batch = cleaned[start : start + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for i, item in enumerate(resp.data):
            out[start + i] = np.asarray(item.embedding, dtype=np.float32)
        cost.log(step, EMBED_MODEL, resp.usage.prompt_tokens, 0)
        done += len(batch)
        if progress_cb:
            progress_cb(done, n)

    return out


def save_embeddings(emb: np.ndarray, review_ids: list, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "embeddings.npz"
    np.savez_compressed(path, embeddings=emb, review_ids=np.asarray(review_ids))
    return path


def load_embeddings(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(out_dir / "embeddings.npz", allow_pickle=False)
    return data["embeddings"], data["review_ids"]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (len(a), len(b)) cosine similarity matrix."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T
