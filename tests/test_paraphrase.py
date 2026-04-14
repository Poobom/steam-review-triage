"""Verification: paraphrased reviews should land in the same cluster, and paraphrased spam should be flagged."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.cache import get_artifact_dir
from src.embed import cosine_similarity_matrix, embed_texts, load_embeddings
from src.llm import CostTracker
from src.spam import (
    NEGATIVE_PROTOTYPES,
    POSITIVE_PROTOTYPES,
    URL_PATTERN,
)

load_dotenv()
INPUT = Path("data/reviews.csv")
OUT = get_artifact_dir(INPUT)


def _load_assignments() -> dict[int, int]:
    payload = json.loads((OUT / "03_clusters.json").read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in payload["assignments"].items()}


def _cluster_centroids(embeddings: np.ndarray, review_ids: np.ndarray, assignments: dict[int, int]) -> dict[int, np.ndarray]:
    rid_to_idx = {int(r): i for i, r in enumerate(review_ids)}
    by_cluster: dict[int, list[int]] = {}
    for rid, cid in assignments.items():
        if rid in rid_to_idx:
            by_cluster.setdefault(cid, []).append(rid_to_idx[rid])
    return {cid: embeddings[idx_list].mean(axis=0) for cid, idx_list in by_cluster.items()}


def _topic_lookup() -> dict[int, str]:
    payload = json.loads((OUT / "04_topics.json").read_text(encoding="utf-8"))
    return {t["cluster_id"]: f'{t["topic"]} / {t["topic_en"]}' for t in payload["topics"]}


def test_paraphrase_clustering():
    """Paraphrased reviews on the same topic should be assigned to the same cluster."""
    embeddings, review_ids = load_embeddings(OUT)
    assignments = _load_assignments()
    centroids = _cluster_centroids(embeddings, review_ids, assignments)
    topics = _topic_lookup()

    test_groups = [
        ("graphics", [
            "The graphics in this game are absolutely stunning.",
            "The visuals are breathtaking and gorgeous.",
            "이 게임의 그래픽이 정말 아름답고 디테일하다.",
        ]),
        ("story", [
            "The story is deeply emotional and well-written.",
            "I was completely captivated by the narrative.",
            "스토리가 너무 몰입감 있고 감동적이었다.",
        ]),
        ("bugs", [
            "The game crashes constantly and ruins the experience.",
            "Game-breaking bugs everywhere, frustrating.",
            "버그가 너무 많아서 진행이 안 된다.",
        ]),
    ]

    cost = CostTracker()
    centroid_ids = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in centroid_ids])

    print("\n=== PARAPHRASE CLUSTERING TEST ===")
    for group_label, sentences in test_groups:
        emb = embed_texts(sentences, cost, step="paraphrase_test")
        sims = cosine_similarity_matrix(emb, centroid_matrix)
        assigned = [centroid_ids[int(i)] for i in sims.argmax(axis=1)]
        same = len(set(assigned)) == 1
        flag = "✓" if same else "✗"
        print(f"{flag} group [{group_label}] -> clusters {assigned}")
        for s, cid in zip(sentences, assigned):
            print(f"   • [c{cid}: {topics.get(cid, '?')[:50]}] {s}")

    print(f"\nCost for this test: ${cost.total:.6f}")


def test_paraphrased_spam_detection():
    """Inject paraphrased spam into the real 4982-row dataset and check detect_spam catches them."""
    import pandas as pd

    from src.clean import load_clean_artifact
    from src.spam import detect_spam

    cost = CostTracker()
    df_orig, _ = load_clean_artifact(OUT)
    embeddings_orig, _ = load_embeddings(OUT)

    paraphrased_spam = [
        "체이크한 CS 스킨을 매우 싸게 판매합니다, 메시지 주세요",
        "공짜 V-bucks 제너레이터 클릭하세요, 한정 기간 이벤트",
        "아이템 거래 디스코드 fake999 최저가 신뢰가능",
        "내 프로필 방문하면 게임 키 무료 증정 이벤트 진행 중",
        "이 게임 후원받고 작성한 광고 리뷰입니다 VPN 서비스 추천",
    ]
    paraphrased_emb = embed_texts(paraphrased_spam, cost, step="paraphrase_test")

    inject_df = pd.DataFrame(
        {
            "review_id": list(range(900001, 900001 + len(paraphrased_spam))),
            "text": paraphrased_spam,
            "lang": ["koreana"] * len(paraphrased_spam),
            "recommended": [True] * len(paraphrased_spam),
            "playtime_hours": [0.0] * len(paraphrased_spam),
            "posted_at": [""] * len(paraphrased_spam),
        }
    )
    df_combined = pd.concat([df_orig, inject_df], ignore_index=True)
    emb_combined = np.vstack([embeddings_orig, paraphrased_emb])

    flags, stats = detect_spam(df_combined, emb_combined, cost)

    print("\n=== SPAM PARAPHRASE DETECTION TEST (injected into real dataset) ===")
    print(f"Threshold (auto-tuned on 5000+ rows): pos>={stats.threshold_pos:.3f}, diff>={stats.threshold_diff:.3f}")
    print(f"Total flagged spam: {stats.spam_count} / {stats.total} ({stats.spam_ratio*100:.2f}%)")

    correct = 0
    inject_ids = set(inject_df["review_id"].tolist())
    inject_flags = flags[flags["review_id"].isin(inject_ids)]
    for text, row in zip(paraphrased_spam, inject_flags.itertuples()):
        actual = "SPAM" if row.is_spam else "LEGIT"
        flag = "✓" if actual == "SPAM" else "✗"
        if actual == "SPAM":
            correct += 1
        print(f"{flag} [{actual}] sim_pos={row.sim_pos:.3f} sim_neg={row.sim_neg:.3f} score={row.spam_score:.3f} :: {text[:60]}")
    print(f"\nDetection rate on injected paraphrases: {correct}/{len(paraphrased_spam)}")
    print(f"Cost for this test: ${cost.total:.6f}")


if __name__ == "__main__":
    test_paraphrase_clustering()
    test_paraphrased_spam_detection()
