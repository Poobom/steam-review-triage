"""Stage 5: Extract actionable negative-review reasons per cluster (gpt-4o-mini)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cluster import _mmr_select
from src.embed import cosine_similarity_matrix
from src.llm import CostTracker, chat_json, get_client

MODEL = "gpt-4o-mini"
MIN_NEGATIVE_FOR_LLM = 5  # below this, skip LLM and just count
SAMPLE_PER_CLUSTER = 20  # nearest 10 + MMR 10
NEAREST_HALF = 10

SYSTEM_PROMPT = (
    "당신은 게임 개발팀에 전달할 사용자 불만 보고서를 작성하는 분석가입니다. "
    "주어진 부정 리뷰에서 공통적으로 제기된 불만 이유를 1~5개 추출하고, "
    "개발팀이 우선순위를 매길 수 있도록 심각도와 대표 인용을 함께 제공합니다. "
    "절대 환각하지 말고 원문에 명시된 정보만 사용하세요. "
    "재현 조건이나 대상(특정 퀘스트, 특정 GPU 등)이 명시되어 있으면 반드시 포함하세요. "
    "출력은 반드시 유효한 JSON 객체여야 합니다."
)

JSON_SCHEMA_HINT = """
응답 JSON 스키마:
{
  "issues": [
    {
      "title": "한국어 이슈 제목 (30자 이내, 구체적)",
      "title_en": "English issue title (concise)",
      "description": "한국어 상세 설명 (2~3 문장, 재현 조건 포함)",
      "description_en": "English description (2~3 sentences)",
      "severity": "critical | major | minor",
      "frequency": 정수 (이 이유를 언급한 리뷰 추정 수),
      "representative_quote_ids": [리뷰_id, ...]
    }
  ]
}
- 빈도 순 + 심각도 순으로 정렬
- severity 기준: critical=게임 진행/실행 차단, major=경험 크게 저해, minor=사소한 불편
- representative_quote_ids: 입력 리뷰 중 가장 대표적인 2~3개 review_id
- 불만이 1개도 없으면 issues=[] 반환
"""


def _build_user_prompt(cluster_id: int, topic: str, reviews: list[dict]) -> str:
    lines = [
        f"# 클러스터: {topic} (id={cluster_id})",
        f"# 부정 리뷰 {len(reviews)}건 샘플",
        "",
        "## 부정 리뷰",
    ]
    for r in reviews:
        text = (r.get("text") or "").replace("\n", " ").strip()
        if len(text) > 350:
            text = text[:350] + "…"
        lines.append(f"- [id={r['review_id']}, {r.get('lang', '?')}] {text}")
    lines.append("")
    lines.append(JSON_SCHEMA_HINT)
    return "\n".join(lines)


def _sample_negative_reviews(
    cluster_member_global_idx: np.ndarray,
    embeddings: np.ndarray,
    n_total: int = SAMPLE_PER_CLUSTER,
) -> list[int]:
    """Sample a diverse set of indices into cluster_member_global_idx for LLM input."""
    n = len(cluster_member_global_idx)
    if n <= n_total:
        return list(range(n))

    cluster_emb = embeddings[cluster_member_global_idx]
    centroid = cluster_emb.mean(axis=0, keepdims=True)
    sim = cosine_similarity_matrix(cluster_emb, centroid).flatten()
    nearest_local = np.argsort(-sim)[:NEAREST_HALF].tolist()
    mmr_local = _mmr_select(cluster_emb, exclude=set(nearest_local), n_select=n_total - NEAREST_HALF)
    return nearest_local + mmr_local


def extract_negative_reasons(
    df_clean: pd.DataFrame,
    embeddings: np.ndarray,
    review_ids: np.ndarray,
    assignments: np.ndarray,
    topics: list[dict],
    cost: CostTracker,
    progress_cb=None,
) -> list[dict]:
    """Per cluster, extract issues from `recommended=False` reviews."""
    client = get_client()
    df_indexed = df_clean.set_index("review_id")
    rid_to_idx = {int(r): i for i, r in enumerate(review_ids)}

    results: list[dict] = []
    total = len(topics)

    for ti, topic in enumerate(topics):
        cid = topic["cluster_id"]
        member_global_idx = np.where(assignments == cid)[0]
        member_rids = [int(review_ids[gi]) for gi in member_global_idx]

        neg_pairs = []
        for rid, gi in zip(member_rids, member_global_idx):
            if rid in df_indexed.index:
                row = df_indexed.loc[rid]
                if not bool(row.get("recommended", False)):
                    neg_pairs.append((rid, gi))

        neg_count = len(neg_pairs)
        cluster_result = {
            "cluster_id": cid,
            "topic": topic["topic"],
            "topic_en": topic["topic_en"],
            "negative_review_count": neg_count,
            "issues": [],
        }

        if neg_count < MIN_NEGATIVE_FOR_LLM:
            results.append(cluster_result)
            if progress_cb:
                progress_cb(ti + 1, total)
            continue

        neg_global_idx = np.array([gi for _rid, gi in neg_pairs])
        local_picks = _sample_negative_reviews(neg_global_idx, embeddings)
        sampled_rids = [int(neg_pairs[lp][0]) for lp in local_picks]

        sample_reviews = []
        for rid in sampled_rids:
            row = df_indexed.loc[rid]
            sample_reviews.append(
                {
                    "review_id": rid,
                    "text": row["text"],
                    "lang": row.get("lang", ""),
                }
            )

        user = _build_user_prompt(cid, topic["topic"], sample_reviews)
        try:
            data = chat_json(
                client=client,
                model=MODEL,
                system=SYSTEM_PROMPT,
                user=user,
                cost=cost,
                step="negative_reasons",
                max_tokens=1500,
                temperature=0.2,
            )
        except Exception as exc:
            data = {"issues": [{"title": "LLM 호출 실패", "title_en": "LLM failure", "description": str(exc), "description_en": str(exc), "severity": "minor", "frequency": 0, "representative_quote_ids": []}]}

        for issue in data.get("issues", []) or []:
            quote_ids = issue.get("representative_quote_ids", []) or []
            quotes = []
            for qid in quote_ids:
                try:
                    qid_int = int(qid)
                    if qid_int in df_indexed.index:
                        quotes.append({"review_id": qid_int, "text": df_indexed.loc[qid_int, "text"]})
                except (ValueError, TypeError):
                    continue
            cluster_result["issues"].append(
                {
                    "title": issue.get("title", "(미정)"),
                    "title_en": issue.get("title_en", "(unknown)"),
                    "description": issue.get("description", ""),
                    "description_en": issue.get("description_en", ""),
                    "severity": issue.get("severity", "minor"),
                    "frequency": int(issue.get("frequency", 0) or 0),
                    "representative_quotes": quotes,
                }
            )

        results.append(cluster_result)
        if progress_cb:
            progress_cb(ti + 1, total)

    return results


def save_negatives_artifact(negatives: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "05_negatives.json"
    path.write_text(json.dumps({"clusters": negatives}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_negatives_artifact(out_dir: Path) -> list[dict]:
    payload = json.loads((out_dir / "05_negatives.json").read_text(encoding="utf-8"))
    return payload["clusters"]
