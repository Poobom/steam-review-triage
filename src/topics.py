"""Stage 4: Name each cluster + bilingual summary using gpt-4o-mini in JSON mode."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.cluster import ClusterInfo
from src.llm import CostTracker, chat_json, get_client

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "당신은 게임 리뷰를 정리하는 데이터 분석가입니다. "
    "동일 주제로 묶인 Steam 리뷰들을 보고, 주제명과 핵심 요약을 한국어와 영어로 모두 작성하세요. "
    "절대 환각하지 말고 주어진 리뷰에서 명시된 정보만 사용하세요. "
    "출력은 반드시 유효한 JSON 객체여야 합니다."
)

JSON_SCHEMA_HINT = """
응답 JSON 스키마:
{
  "topic": "한국어 주제명 (20자 이내)",
  "topic_en": "English topic name (concise, 6 words max)",
  "summary": ["요약 1", "요약 2", "요약 3"],
  "summary_en": ["summary 1", "summary 2", "summary 3"],
  "sentiment": "positive | negative | mixed",
  "keywords_ko": ["키워드", "..."],
  "keywords_en": ["keyword", "..."],
  "representative_quote_ids": [리뷰_id, ...]
}
- summary 는 정확히 3줄, 각 줄은 한 문장
- representative_quote_ids 는 입력으로 준 review_id 중 가장 대표적인 2~3개 선택
- "기타" 클러스터(id=-1)인 경우에도 동일한 형식으로 답하되 주제명에 "기타·다양"을 포함
"""


def _build_user_prompt(cluster_id: int, reviews: list[dict]) -> str:
    lines = [
        f"# 클러스터 ID: {cluster_id}",
        f"# 리뷰 수: {len(reviews)}건 중 대표 {min(len(reviews), 10)}건",
        "",
        "## 리뷰 목록",
    ]
    for r in reviews:
        lang = r.get("lang", "?")
        rec = "👍추천" if r.get("recommended") else "👎비추천"
        text = (r.get("text") or "").replace("\n", " ").strip()
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"- [id={r['review_id']}, {lang}, {rec}] {text}")
    lines.append("")
    lines.append(JSON_SCHEMA_HINT)
    return "\n".join(lines)


def name_clusters(
    clusters: list[ClusterInfo],
    df_clean: pd.DataFrame,
    cost: CostTracker,
    progress_cb=None,
) -> list[dict]:
    """For each cluster, call LLM with its representative reviews. Returns list of topic dicts."""
    client = get_client()
    df_indexed = df_clean.set_index("review_id")
    results: list[dict] = []

    for i, cluster in enumerate(clusters):
        rep_rows = []
        for rid in cluster.representatives:
            if rid in df_indexed.index:
                row = df_indexed.loc[rid]
                rep_rows.append(
                    {
                        "review_id": int(rid),
                        "text": row["text"],
                        "lang": row.get("lang", ""),
                        "recommended": bool(row.get("recommended", False)),
                    }
                )

        user = _build_user_prompt(cluster.id, rep_rows)
        try:
            data = chat_json(
                client=client,
                model=MODEL,
                system=SYSTEM_PROMPT,
                user=user,
                cost=cost,
                step="cluster_naming",
                max_tokens=900,
                temperature=0.2,
            )
        except Exception as exc:
            data = {
                "topic": f"클러스터 {cluster.id}",
                "topic_en": f"Cluster {cluster.id}",
                "summary": [f"LLM 호출 실패: {exc}"],
                "summary_en": [f"LLM call failed: {exc}"],
                "sentiment": "mixed",
                "keywords_ko": [],
                "keywords_en": [],
                "representative_quote_ids": cluster.representatives[:3],
            }

        quote_ids = data.get("representative_quote_ids", []) or cluster.representatives[:3]
        rep_quotes = []
        for qid in quote_ids:
            try:
                qid_int = int(qid)
                if qid_int in df_indexed.index:
                    rep_quotes.append({"review_id": qid_int, "text": df_indexed.loc[qid_int, "text"]})
            except (ValueError, TypeError):
                continue

        results.append(
            {
                "cluster_id": int(cluster.id),
                "size": int(cluster.size),
                "topic": data.get("topic", "(미정)"),
                "topic_en": data.get("topic_en", "(unknown)"),
                "summary": data.get("summary", []),
                "summary_en": data.get("summary_en", []),
                "sentiment": data.get("sentiment", "mixed"),
                "keywords_ko": data.get("keywords_ko", []),
                "keywords_en": data.get("keywords_en", []),
                "representative_quotes": rep_quotes,
            }
        )
        if progress_cb:
            progress_cb(i + 1, len(clusters))

    return results


def save_topics_artifact(topics: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "04_topics.json"
    path.write_text(json.dumps({"topics": topics}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_topics_artifact(out_dir: Path) -> list[dict]:
    payload = json.loads((out_dir / "04_topics.json").read_text(encoding="utf-8"))
    return payload["topics"]
