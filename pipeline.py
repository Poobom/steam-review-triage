"""End-to-end pipeline: 6 stages with disk cache by input SHA1.

CLI usage:
    python pipeline.py --input data/reviews.csv [--force]

Programmatic usage (called by app.py):
    from pipeline import run_pipeline
    result = run_pipeline(Path("data/reviews.csv"), on_step=lambda i, label, info: ...)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv

from src.cache import get_artifact_dir, stage_done
from src.clean import (
    load_and_clean,
    load_clean_artifact,
    save_clean_artifact,
)
from src.cluster import (
    load_cluster_artifact,
    run_clustering,
    save_cluster_artifact,
)
from src.embed import embed_texts, load_embeddings, save_embeddings
from src.export import (
    generate_jira_payload,
    generate_markdown,
    save_export_artifacts,
)
from src.llm import CostTracker
from src.negatives import (
    extract_negative_reasons,
    load_negatives_artifact,
    save_negatives_artifact,
)
from src.spam import (
    detect_spam,
    load_spam_artifact,
    save_spam_artifact,
)
from src.topics import (
    load_topics_artifact,
    name_clusters,
    save_topics_artifact,
)

load_dotenv()

STAGES = [
    (1, "정제 및 인코딩 복구", "01_clean.json"),
    (2, "임베딩 및 스팸 탐지", "02_spam.json"),
    (3, "의미 기반 클러스터링", "03_clusters.json"),
    (4, "클러스터 주제 명명", "04_topics.json"),
    (5, "부정 이유 추출", "05_negatives.json"),
    (6, "리포트·Jira JSON 생성", "06_report.md"),
]


def _noop(*_args, **_kwargs) -> None:
    return None


def run_pipeline(
    input_path: Path,
    on_step: Callable[[int, str, dict], None] | None = None,
    force: bool = False,
) -> dict:
    """Execute 6 stages with caching. on_step(idx, label, info_dict) called per stage."""
    on_step = on_step or _noop
    out_dir = get_artifact_dir(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = CostTracker()

    cache_hits = 0

    # --- Stage 1: Clean ---
    on_step(1, STAGES[0][1], {"status": "start"})
    if not force and stage_done(out_dir, STAGES[0][2]):
        df_clean, clean_stats = load_clean_artifact(out_dir)
        cache_hits += 1
    else:
        df_clean, clean_stats = load_and_clean(input_path)
        save_clean_artifact(df_clean, clean_stats, out_dir)
    on_step(1, STAGES[0][1], {
        "status": "done",
        "input": clean_stats.input_count,
        "output": clean_stats.output_count,
        "encoding_repaired": clean_stats.encoding_repaired,
    })

    # --- Stage 2: Embed + Spam ---
    on_step(2, STAGES[1][1], {"status": "start"})
    emb_path = out_dir / "embeddings.npz"
    if not force and emb_path.exists():
        embeddings, review_ids = load_embeddings(out_dir)
    else:
        texts = df_clean["text"].tolist()
        embeddings = embed_texts(texts, cost, step="embedding")
        review_ids = df_clean["review_id"].to_numpy()
        save_embeddings(embeddings, review_ids.tolist(), out_dir)

    if not force and stage_done(out_dir, STAGES[1][2]):
        spam_flags, spam_stats = load_spam_artifact(out_dir)
        cache_hits += 1
    else:
        spam_flags, spam_stats = detect_spam(df_clean, embeddings, cost)
        save_spam_artifact(spam_flags, spam_stats, out_dir)
    on_step(2, STAGES[1][1], {
        "status": "done",
        "embedded": int(len(embeddings)),
        "spam": spam_stats.spam_count,
        "spam_ratio": round(spam_stats.spam_ratio, 4),
    })

    # --- Stage 3: Cluster (on non-spam subset) ---
    on_step(3, STAGES[2][1], {"status": "start"})
    spam_set = set(spam_flags.loc[spam_flags["is_spam"], "review_id"].tolist())
    keep_mask = ~df_clean["review_id"].isin(spam_set).to_numpy()
    df_active = df_clean.loc[keep_mask].reset_index(drop=True)
    emb_active = embeddings[keep_mask]
    rid_active = df_active["review_id"].to_numpy()

    if not force and stage_done(out_dir, STAGES[2][2]):
        assignments_dict, clusters, cluster_stats = load_cluster_artifact(out_dir)
        assignments = np.array([assignments_dict.get(int(rid), -1) for rid in rid_active])
        cache_hits += 1
    else:
        assignments, clusters, cluster_stats = run_clustering(emb_active, rid_active)
        save_cluster_artifact(assignments, clusters, cluster_stats, rid_active, out_dir)
    on_step(3, STAGES[2][1], {
        "status": "done",
        "k": cluster_stats.k,
        "silhouette": round(cluster_stats.silhouette, 4),
        "other": cluster_stats.other_count,
    })

    # --- Stage 4: Cluster Naming ---
    on_step(4, STAGES[3][1], {"status": "start"})
    if not force and stage_done(out_dir, STAGES[3][2]):
        topics = load_topics_artifact(out_dir)
        cache_hits += 1
    else:
        topics = name_clusters(clusters, df_active, cost)
        save_topics_artifact(topics, out_dir)
    on_step(4, STAGES[3][1], {"status": "done", "topics": len(topics)})

    # --- Stage 5: Negative Reasons ---
    on_step(5, STAGES[4][1], {"status": "start"})
    if not force and stage_done(out_dir, STAGES[4][2]):
        negatives = load_negatives_artifact(out_dir)
        cache_hits += 1
    else:
        negatives = extract_negative_reasons(df_active, emb_active, rid_active, assignments, topics, cost)
        save_negatives_artifact(negatives, out_dir)
    total_issues = sum(len(n.get("issues", [])) for n in negatives)
    on_step(5, STAGES[4][1], {"status": "done", "issues": total_issues})

    # --- Stage 6: Export ---
    on_step(6, STAGES[5][1], {"status": "start"})
    md_text = generate_markdown(
        clean_stats={"input_count": clean_stats.input_count, "output_count": clean_stats.output_count, "encoding_repaired": clean_stats.encoding_repaired},
        spam_stats={"spam_count": spam_stats.spam_count, "spam_ratio": spam_stats.spam_ratio},
        cluster_stats={"k": cluster_stats.k, "silhouette": cluster_stats.silhouette},
        topics=topics,
        negatives=negatives,
        total_cost=cost.total,
    )
    jira_payload = generate_jira_payload(topics, negatives)
    save_export_artifacts(md_text, jira_payload, out_dir)
    cost.save(out_dir)
    on_step(6, STAGES[5][1], {"status": "done", "cost_usd": round(cost.total, 6)})

    return {
        "out_dir": str(out_dir),
        "cache_hits": cache_hits,
        "clean_stats": clean_stats,
        "spam_stats": spam_stats,
        "cluster_stats": cluster_stats,
        "topics": topics,
        "negatives": negatives,
        "cost_total": cost.total,
        "cost_breakdown": cost.by_step(),
    }


def _cli():
    parser = argparse.ArgumentParser(description="Steam review classification pipeline")
    parser.add_argument("--input", default="data/reviews.csv", help="Input CSV path")
    parser.add_argument("--force", action="store_true", help="Ignore cache, rerun all stages")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    def log(idx, label, info):
        ts = time.strftime("%H:%M:%S")
        info_str = ", ".join(f"{k}={v}" for k, v in info.items() if k != "status")
        status = info.get("status", "?")
        print(f"[{ts}] [{idx}/6] {label} :: {status} {info_str}".rstrip())

    result = run_pipeline(input_path, on_step=log, force=args.force)
    print("\n=== SUMMARY ===")
    print(f"Output dir   : {result['out_dir']}")
    print(f"Cache hits   : {result['cache_hits']}/6 stages")
    print(f"Total cost   : ${result['cost_total']:.4f}")
    print(f"Topics       : {len(result['topics'])}")
    print(f"Issues total : {sum(len(n.get('issues', [])) for n in result['negatives'])}")


if __name__ == "__main__":
    _cli()
