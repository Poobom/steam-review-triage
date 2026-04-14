"""Stage 6: Export Markdown report and Jira bulk-create JSON from earlier artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SEVERITY_TO_PRIORITY = {"critical": "Highest", "major": "High", "minor": "Medium"}
SENTIMENT_BADGE = {"positive": "🟢 긍정", "negative": "🔴 부정", "mixed": "🟡 혼합"}


def _kpi_section(clean_stats: dict, spam_stats: dict, cluster_stats: dict, total_negative: int, total_cost: float) -> str:
    return f"""## 📊 KPI 요약

| 지표 | 값 |
|---|---|
| 분석 일시 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |
| 입력 리뷰 | {clean_stats.get('input_count', '?'):,}건 |
| 정상 리뷰 | {clean_stats.get('output_count', '?') - spam_stats.get('spam_count', 0):,}건 |
| 스팸 후보 | {spam_stats.get('spam_count', '?'):,}건 ({spam_stats.get('spam_ratio', 0)*100:.1f}%) |
| 인코딩 복구 | {clean_stats.get('encoding_repaired', '?'):,}건 |
| 클러스터 수 | {cluster_stats.get('k', '?')} (silhouette={cluster_stats.get('silhouette', 0):.3f}) |
| 부정 리뷰 | {total_negative:,}건 |
| LLM 비용 | ${total_cost:.4f} |
"""


def _topic_section(topic: dict, neg_lookup: dict) -> str:
    cid = topic["cluster_id"]
    badge = SENTIMENT_BADGE.get(topic.get("sentiment", "mixed"), "")
    summary_lines = "\n".join(f"- {s}" for s in topic.get("summary", []))
    summary_en_lines = "\n".join(f"- {s}" for s in topic.get("summary_en", []))
    keywords = " · ".join(topic.get("keywords_ko", []) or [])
    keywords_en = " · ".join(topic.get("keywords_en", []) or [])

    quotes_md = ""
    for q in topic.get("representative_quotes", [])[:3]:
        text = (q.get("text") or "").replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200] + "…"
        quotes_md += f"> {text} _— review #{q.get('review_id')}_\n\n"

    neg_md = ""
    cluster_neg = neg_lookup.get(cid)
    if cluster_neg and cluster_neg.get("issues"):
        neg_md += "\n#### 🚨 주요 불만 이유\n\n"
        neg_md += "| # | 심각도 | 빈도 | 이슈 (KO) | 이슈 (EN) | 대표 리뷰 |\n"
        neg_md += "|---|---|---|---|---|---|\n"
        for i, issue in enumerate(cluster_neg["issues"], 1):
            sev = issue.get("severity", "minor")
            freq = issue.get("frequency", 0)
            title = issue.get("title", "").replace("|", "/")
            title_en = issue.get("title_en", "").replace("|", "/")
            rep_ids = [str(q["review_id"]) for q in issue.get("representative_quotes", [])[:2]]
            neg_md += f"| {i} | {sev} | {freq} | {title} | {title_en} | {', '.join(rep_ids)} |\n"

    return f"""---

### 클러스터 {cid} — {topic.get('topic', '?')} ({topic.get('topic_en', '?')})
**리뷰 수**: {topic.get('size', '?'):,} / **감정**: {badge}
**키워드**: {keywords}
**Keywords**: {keywords_en}

**요약**
{summary_lines}

**Summary**
{summary_en_lines}

**대표 인용**

{quotes_md}{neg_md}
"""


def generate_markdown(
    clean_stats: dict,
    spam_stats: dict,
    cluster_stats: dict,
    topics: list[dict],
    negatives: list[dict],
    total_cost: float,
) -> str:
    neg_lookup = {n["cluster_id"]: n for n in negatives}
    total_negative = sum(n.get("negative_review_count", 0) for n in negatives)

    sections = ["# Steam 리뷰 분석 리포트\n"]
    sections.append(_kpi_section(clean_stats, spam_stats, cluster_stats, total_negative, total_cost))
    sections.append("\n## 🎯 주제별 분석\n")

    sorted_topics = sorted(topics, key=lambda t: -t.get("size", 0))
    for topic in sorted_topics:
        sections.append(_topic_section(topic, neg_lookup))

    return "\n".join(sections)


def generate_jira_payload(
    topics: list[dict],
    negatives: list[dict],
    project_key: str = "GAME",
) -> dict:
    """Build Jira bulk-create payload. Only negative cluster issues become Jira tickets."""
    topic_lookup = {t["cluster_id"]: t for t in topics}
    issue_updates: list[dict] = []

    for neg in negatives:
        cid = neg["cluster_id"]
        topic = topic_lookup.get(cid, {})
        topic_name = topic.get("topic", f"Cluster {cid}")
        topic_name_en = topic.get("topic_en", "")

        for issue in neg.get("issues", []):
            severity = issue.get("severity", "minor")
            priority = SEVERITY_TO_PRIORITY.get(severity, "Medium")
            quotes = issue.get("representative_quotes", [])
            quote_text = "\n\n대표 인용:\n" + "\n".join(
                f'- "{(q.get("text") or "").replace(chr(10), " ").strip()[:300]}" (review #{q.get("review_id")})'
                for q in quotes[:3]
            ) if quotes else ""

            description = (
                f"클러스터: {topic_name} ({topic_name_en})\n"
                f"클러스터 부정 리뷰: {neg.get('negative_review_count', 0)}건\n"
                f"이 이슈 추정 빈도: {issue.get('frequency', 0)}건\n\n"
                f"{issue.get('description', '')}\n\n"
                f"--- English ---\n{issue.get('description_en', '')}"
                f"{quote_text}"
            )

            issue_updates.append(
                {
                    "fields": {
                        "project": {"key": project_key},
                        "issuetype": {"name": "Bug"},
                        "summary": f"[CX/리뷰분석] {issue.get('title', '제목 없음')}",
                        "description": description,
                        "priority": {"name": priority},
                        "labels": ["cx-review", "auto-extracted", f"cluster-{cid}", f"severity-{severity}"],
                    }
                }
            )

    return {"issueUpdates": issue_updates}


def save_export_artifacts(
    markdown: str,
    jira_payload: dict,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "06_report.md"
    jira_path = out_dir / "06_jira.json"
    md_path.write_text(markdown, encoding="utf-8")
    jira_path.write_text(json.dumps(jira_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, jira_path
