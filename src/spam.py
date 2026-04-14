"""Stage 2b: Bipolar zero-shot spam detection using prototype similarity + URL marker."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.embed import cosine_similarity_matrix, embed_texts
from src.llm import CostTracker

POSITIVE_PROTOTYPES = [
    "외부 사이트로 유도하는 광고 또는 프로모션 메시지로 구매를 권유함",
    "게임 아이템·계정·재화의 현금거래를 알선하거나 카톡·연락처로 거래 유도",
    "URL이나 외부 링크가 본문 대부분을 차지하는 홍보성 짧은 글",
    "유료 후원, 스폰서, 광고비를 받고 작성된 리뷰임을 명시하는 문장",
    "봇이 자동 생성한 의미 없는 반복 텍스트, 이모지 도배, 무관한 키워드 나열",
]

NEGATIVE_PROTOTYPES = [
    "게임을 직접 플레이해 본 경험에 기반한 솔직한 감상과 평가",
    "게임의 그래픽, 스토리, 캐릭터, 시스템, 난이도 등 구체적 요소에 대한 의견",
    "게임을 추천하거나 비추천하는 짧지만 진정성 있는 한 줄 평",
]

URL_PATTERN = re.compile(
    r"(?:https?://|www\.|bit\.ly|t\.co|tinyurl|[a-z0-9-]+\.(?:com|net|org|io|gg|kr|cn|ru|info|biz)\b)",
    re.IGNORECASE,
)


@dataclass
class SpamStats:
    total: int
    spam_count: int
    spam_ratio: float
    threshold_pos: float
    threshold_diff: float
    url_boost_count: int


def _percentile_threshold(scores: np.ndarray, target_ratio: float = 0.05) -> float:
    """Return threshold such that ~target_ratio of scores exceed it."""
    if len(scores) == 0:
        return 1.0
    return float(np.quantile(scores, 1.0 - target_ratio))


def detect_spam(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cost: CostTracker,
    target_spam_ratio: float = 0.06,
    min_pos_threshold: float = 0.32,
    min_diff_threshold: float = 0.03,
) -> tuple[pd.DataFrame, SpamStats]:
    """Compute spam flags using positive/negative prototype similarity + URL marker.

    Returns (flags_df with columns [review_id, is_spam, spam_score, sim_pos, sim_neg, has_url, matched_prototype], stats).
    """
    pos_emb = embed_texts(POSITIVE_PROTOTYPES, cost, step="embedding_prototypes")
    neg_emb = embed_texts(NEGATIVE_PROTOTYPES, cost, step="embedding_prototypes")

    sim_pos_full = cosine_similarity_matrix(embeddings, pos_emb)  # (n, 5)
    sim_neg_full = cosine_similarity_matrix(embeddings, neg_emb)  # (n, 3)

    sim_pos = sim_pos_full.max(axis=1)
    matched_idx = sim_pos_full.argmax(axis=1)
    sim_neg = sim_neg_full.max(axis=1)
    diff = sim_pos - sim_neg

    has_url = df["text"].str.contains(URL_PATTERN, regex=True, na=False).to_numpy()
    spam_score = diff + np.where(has_url, 0.10, 0.0)

    pos_thresh_data = _percentile_threshold(sim_pos, target_spam_ratio)
    pos_thresh = max(pos_thresh_data, min_pos_threshold)

    is_spam = (sim_pos >= pos_thresh) & (diff >= min_diff_threshold)
    is_spam = is_spam | has_url

    flags = pd.DataFrame(
        {
            "review_id": df["review_id"].values,
            "is_spam": is_spam,
            "spam_score": np.round(spam_score, 4),
            "sim_pos": np.round(sim_pos, 4),
            "sim_neg": np.round(sim_neg, 4),
            "has_url": has_url,
            "matched_prototype": [POSITIVE_PROTOTYPES[i] for i in matched_idx],
        }
    )

    stats = SpamStats(
        total=int(len(df)),
        spam_count=int(is_spam.sum()),
        spam_ratio=float(is_spam.mean()),
        threshold_pos=float(pos_thresh),
        threshold_diff=float(min_diff_threshold),
        url_boost_count=int(has_url.sum()),
    )
    return flags, stats


def save_spam_artifact(flags: pd.DataFrame, stats: SpamStats, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "02_spam.json"
    payload = {
        "stats": asdict(stats),
        "prototypes": {"positive": POSITIVE_PROTOTYPES, "negative": NEGATIVE_PROTOTYPES},
        "flags": flags.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_spam_artifact(out_dir: Path) -> tuple[pd.DataFrame, SpamStats]:
    payload = json.loads((out_dir / "02_spam.json").read_text(encoding="utf-8"))
    flags = pd.DataFrame(payload["flags"])
    stats = SpamStats(**payload["stats"])
    return flags, stats
