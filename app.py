"""Streamlit UI for Task 01 — Steam 리뷰 분류·요약 파이프라인.

CX팀 매니저(비개발자)가 크롬에서 URL 로 바로 접속해 5,000건 Steam 리뷰를
주제별로 묶어 보고 부정 리뷰 이유를 개발팀에 전달할 수 있도록 하는 앱.

백엔드 파이프라인(`pipeline.py`)과 독립적으로 작동하도록 설계되었으며,
pipeline import 가 실패하면 데모 모드로 전환해 precomputed artifacts 를 표시한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

# ---------------------------------------------------------------------------
# 상수 / 경로
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DEFAULT = ROOT / "data" / "reviews.csv"
ARTIFACTS_ROOT = ROOT / "artifacts"
REQUIRED_COLUMNS = {"review_id", "text", "lang", "recommended", "playtime_hours", "posted_at"}

STAGE_LABELS = [
    "1/6 · 로딩 & 인코딩 복구",
    "2/6 · 스팸 탐지",
    "3/6 · 클러스터링",
    "4/6 · 주제 요약",
    "5/6 · 부정 이유 추출",
    "6/6 · 리포트 생성",
]

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}
SEVERITY_LABEL = {"critical": "🔴 Critical", "major": "🟠 Major", "minor": "🟡 Minor"}

# pipeline import 는 실패해도 앱은 살아있어야 한다 (데모 모드)
try:
    from pipeline import run_pipeline  # type: ignore

    PIPELINE_AVAILABLE = True
    PIPELINE_IMPORT_ERROR = None
except Exception as exc:  # ImportError 혹은 backend 로딩 중 에러
    run_pipeline = None  # type: ignore
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# 유틸: 해시 / artifact I/O
# ---------------------------------------------------------------------------
def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def artifacts_dir(sha1: str) -> Path:
    return ARTIFACTS_ROOT / sha1


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def load_artifacts(sha1: str) -> dict[str, Any] | None:
    """SHA1 해시 폴더에서 artifact 번들을 로드한다. 필수 파일이 빠지면 None."""
    d = artifacts_dir(sha1)
    if not d.exists():
        return None
    bundle = {
        "clean": load_json(d / "01_clean.json"),
        "spam": load_json(d / "02_spam.json"),
        "clusters": load_json(d / "03_clusters.json"),
        "topics": load_json(d / "04_topics.json"),
        "negatives": load_json(d / "05_negatives.json"),
        "report_md": load_text(d / "06_report.md"),
        "jira": load_json(d / "06_jira.json"),
        "cost": load_json(d / "cost.json"),
        "sha1": sha1,
        "dir": str(d),
    }
    # 최소 01_clean 은 있어야 의미가 있음
    if bundle["clean"] is None:
        return None
    return bundle


# ---------------------------------------------------------------------------
# 유틸: CSV 검증
# ---------------------------------------------------------------------------
def validate_csv(path: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        return False, f"CSV 읽기 실패: {exc}"
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, f"필수 컬럼 누락: {', '.join(sorted(missing))}"
    return True, "OK"


def save_uploaded_csv(uploaded) -> Path:
    """업로드된 파일을 임시 위치에 저장하고 경로를 반환."""
    tmp_dir = ROOT / ".tmp_uploads"
    tmp_dir.mkdir(exist_ok=True)
    target = tmp_dir / uploaded.name
    target.write_bytes(uploaded.getbuffer())
    return target


# ---------------------------------------------------------------------------
# 파이프라인 실행 (progress UI 연계)
# ---------------------------------------------------------------------------
def run_with_progress(input_path: Path) -> dict[str, Any] | None:
    """pipeline.run_pipeline 를 호출하며 st.progress + st.status 를 갱신한다.

    pipeline 이 import 실패 상태면 캐시된 artifacts 를 5초에 걸쳐 replay 하는
    데모 모드로 동작한다.
    """
    progress = st.progress(0, text="분석을 시작합니다…")
    status = st.status("파이프라인 실행 중", expanded=True)

    def on_step(stage_idx: int, label: str, info: dict | None = None) -> None:
        # pipeline.py 는 1-based stage_idx (1~6) 를 사용. start/done 두 번 호출.
        pct = int((stage_idx / len(STAGE_LABELS)) * 100)
        progress.progress(pct, text=f"[{stage_idx}/6] {label} ({pct}%)")
        if info and info.get("status") == "done":
            with status:
                bits = [f"**{k}**: {v}" for k, v in info.items() if k != "status"]
                detail = " · ".join(bits) if bits else "완료"
                st.write(f"✅ [{stage_idx}/6] {label} — {detail}")
        elif info and info.get("status") == "start":
            with status:
                st.write(f"🔄 [{stage_idx}/6] {label} — 진행 중…")

    try:
        if PIPELINE_AVAILABLE and run_pipeline is not None:
            run_pipeline(input_path, on_step)  # type: ignore[misc]
            # 번들 형태로 통일: pipeline 결과 대신 artifacts 디스크 로드
            sha1 = sha1_of_file(input_path)
            bundle = load_artifacts(sha1)
            if bundle is None:
                status.update(label="결과 로드 실패", state="error", expanded=True)
                st.error("파이프라인은 끝났으나 artifacts 폴더를 찾지 못했습니다.")
                return None
            status.update(label="분석 완료", state="complete", expanded=False)
            progress.progress(100, text="완료")
            return bundle

        # 데모 모드: 캐시 확인 후 replay
        sha1 = sha1_of_file(input_path)
        bundle = load_artifacts(sha1)
        if bundle is None:
            status.update(label="백엔드 미준비", state="error", expanded=True)
            st.error(
                "`pipeline.py` 를 불러올 수 없고, 이 입력에 대한 캐시된 결과도 없습니다.\n"
                "메인 에이전트가 백엔드 구현을 마칠 때까지 기다려주세요."
            )
            if PIPELINE_IMPORT_ERROR:
                st.caption(f"import 에러: {PIPELINE_IMPORT_ERROR}")
            return None

        for i, label in enumerate(STAGE_LABELS):
            time.sleep(0.5)
            on_step(i, label, {"cache": "hit"})
        status.update(label="캐시 replay 완료", state="complete", expanded=False)
        progress.progress(100, text="완료 (캐시)")
        return bundle
    except Exception as exc:
        status.update(label="파이프라인 실패", state="error", expanded=True)
        st.exception(exc)
        return None


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
def render_sidebar(current: dict | None) -> tuple[Path | None, bool]:
    st.sidebar.title("⚙️ 설정")

    mode = st.sidebar.radio(
        "입력 데이터 선택",
        ["기본 reviews.csv (5,000건)", "다른 CSV 업로드"],
        index=0,
    )

    input_path: Path | None = None
    if mode.startswith("기본"):
        if DATA_DEFAULT.exists():
            input_path = DATA_DEFAULT
            st.sidebar.caption(f"📄 `{DATA_DEFAULT.relative_to(ROOT)}`")
        else:
            st.sidebar.error("기본 데이터 파일이 없습니다.")
    else:
        uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
        if uploaded is not None:
            input_path = save_uploaded_csv(uploaded)
            ok, msg = validate_csv(input_path)
            if not ok:
                st.sidebar.error(msg)
                input_path = None
            else:
                st.sidebar.success(f"업로드 확인: {uploaded.name}")

    start = st.sidebar.button(
        "🚀 분석 시작",
        type="primary",
        use_container_width=True,
        disabled=input_path is None,
    )

    st.sidebar.divider()
    st.sidebar.subheader("ℹ️ 모델 / 비용")
    st.sidebar.markdown(
        "- **임베딩**: OpenAI `text-embedding-3-small`\n"
        "- **요약**: OpenAI `gpt-4o-mini`\n"
    )

    run_cost = 0.0
    last_run = "—"
    if current and current.get("cost"):
        cost = current["cost"]
        run_cost = float(cost.get("total_cost_usd", 0.0) or 0.0)
        records = cost.get("records") or []
        if records:
            last_run = records[-1].get("timestamp", "—")
    st.sidebar.metric("이번 실행 비용", f"${run_cost:.4f}")
    st.sidebar.caption(f"마지막 분석: {last_run}")

    # API 키 가용성
    if not _get_api_key():
        st.sidebar.warning("OPENAI_API_KEY 가 설정되어 있지 않습니다. `.env` 또는 Streamlit Secrets 를 확인하세요.")

    if not PIPELINE_AVAILABLE:
        st.sidebar.info("🧪 데모 모드 — `pipeline.py` 가 아직 준비되지 않아 캐시된 결과만 표시합니다.")

    return input_path, start


def _get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# KPI 헤더
# ---------------------------------------------------------------------------
def render_kpis(bundle: dict[str, Any]) -> None:
    clean_stats = (bundle.get("clean") or {}).get("stats", {})
    spam_stats = (bundle.get("spam") or {}).get("stats", {})
    clusters = (bundle.get("clusters") or {}).get("clusters", [])
    cost = bundle.get("cost") or {}

    total = clean_stats.get("output_count") or clean_stats.get("input_count") or 0
    spam_n = spam_stats.get("spam_count", 0)
    repaired = clean_stats.get("encoding_repaired", 0)
    cluster_stats = (bundle.get("clusters") or {}).get("stats", {})
    k = cluster_stats.get("k", len(clusters))
    total_cost = float(cost.get("total_cost_usd", 0.0) or 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 리뷰 수", f"{total:,}")
    c2.metric("스팸 후보", f"{spam_n:,}")
    c3.metric("인코딩 복구", f"{repaired:,}")
    c4.metric("클러스터 수", f"{k}")
    c5.metric("비용 합계", f"${total_cost:.4f}")


# ---------------------------------------------------------------------------
# 탭: 주제별 보기
# ---------------------------------------------------------------------------
def render_topics_tab(bundle: dict[str, Any]) -> None:
    topics_data = bundle.get("topics") or {}
    topics = topics_data.get("topics") if isinstance(topics_data, dict) else topics_data
    if not topics:
        st.info("주제 요약 결과가 없습니다.")
        return

    clusters_meta = {c["id"]: c for c in (bundle.get("clusters") or {}).get("clusters", [])}
    negatives_map = _negatives_by_cluster(bundle)

    col_sort, col_filter = st.columns([1, 1])
    sort_by = col_sort.radio(
        "정렬", ["크기 순", "부정 비율 순"], horizontal=True, key="topics_sort"
    )
    sentiment_filter = col_filter.multiselect(
        "감성 필터",
        ["positive", "negative", "mixed"],
        default=["positive", "negative", "mixed"],
        key="topics_sentiment",
    )

    def size_of(t: dict) -> int:
        cid = t.get("cluster_id")
        return int(clusters_meta.get(cid, {}).get("size", 0))

    def neg_ratio_of(t: dict) -> float:
        cid = t.get("cluster_id")
        size = size_of(t) or 1
        neg = negatives_map.get(cid, {}).get("negative_review_count", 0)
        return neg / size

    items = [t for t in topics if t.get("sentiment", "mixed") in sentiment_filter]
    if sort_by.startswith("크기"):
        items.sort(key=size_of, reverse=True)
    else:
        items.sort(key=neg_ratio_of, reverse=True)

    for topic in items:
        cid = topic.get("cluster_id")
        size = size_of(topic)
        sentiment = topic.get("sentiment", "mixed")
        neg_ratio = neg_ratio_of(topic)
        icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}.get(sentiment, "⚪")
        header = (
            f"{icon} #{cid} · {topic.get('topic', '(제목 없음)')} "
            f"— {size}건 · 부정 {neg_ratio:.0%}"
        )
        with st.expander(header, expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**주제 (한)**: {topic.get('topic', '-')}")
                st.markdown("**요약 (한)**")
                for line in topic.get("summary", []) or []:
                    st.markdown(f"- {line}")
                kws = topic.get("keywords_ko") or []
                if kws:
                    st.caption("키워드: " + ", ".join(kws))
            with c2:
                st.markdown(f"**Topic (EN)**: {topic.get('topic_en', '-')}")
                st.markdown("**Summary (EN)**")
                for line in topic.get("summary_en", []) or []:
                    st.markdown(f"- {line}")
                kws = topic.get("keywords_en") or []
                if kws:
                    st.caption("Keywords: " + ", ".join(kws))

            quotes = topic.get("representative_quotes") or []
            if quotes:
                st.markdown("**대표 인용**")
                for q in quotes[:5]:
                    st.markdown(
                        f"> {q.get('text', '')}  \n"
                        f"> <small>review_id: `{q.get('review_id')}`</small>",
                        unsafe_allow_html=True,
                    )


def _negatives_by_cluster(bundle: dict[str, Any]) -> dict[int, dict]:
    negs = bundle.get("negatives") or []
    if isinstance(negs, dict):
        negs = negs.get("clusters") or []
    return {n.get("cluster_id"): n for n in negs if isinstance(n, dict)}


# ---------------------------------------------------------------------------
# 탭: 부정 이유
# ---------------------------------------------------------------------------
def render_negatives_tab(bundle: dict[str, Any]) -> None:
    negs_by_cluster = _negatives_by_cluster(bundle)
    if not negs_by_cluster:
        st.info("부정 이유 분석 결과가 없습니다.")
        return

    # 평탄화
    rows: list[dict] = []
    for cid, entry in negs_by_cluster.items():
        for idx, issue in enumerate(entry.get("issues", []) or []):
            rows.append(
                {
                    "cluster_id": cid,
                    "topic": entry.get("topic", "-"),
                    "title": issue.get("title", "-"),
                    "title_en": issue.get("title_en", "-"),
                    "severity": issue.get("severity", "minor"),
                    "frequency": int(issue.get("frequency", 0) or 0),
                    "description": issue.get("description", ""),
                    "description_en": issue.get("description_en", ""),
                    "quotes": issue.get("representative_quotes") or [],
                    "_key": f"{cid}-{idx}",
                }
            )
    if not rows:
        st.info("추출된 부정 이슈가 없습니다.")
        return

    df = pd.DataFrame(rows)
    df["sev_rank"] = df["severity"].map(SEVERITY_ORDER).fillna(9).astype(int)
    df = df.sort_values(["sev_rank", "frequency"], ascending=[True, False]).reset_index(drop=True)
    df["심각도"] = df["severity"].map(SEVERITY_LABEL).fillna(df["severity"])

    display_df = df[["cluster_id", "topic", "title", "심각도", "frequency"]].rename(
        columns={"cluster_id": "CID", "topic": "주제", "title": "이슈", "frequency": "빈도"}
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    options = [f"#{r.cluster_id} — {r.title} ({r.severity})" for r in df.itertuples()]
    selected = st.selectbox("상세 보기 (행 선택)", options, index=0 if options else None)
    if selected:
        row = df.iloc[options.index(selected)]
        st.divider()
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"### {row['title']}")
            st.markdown(f"**심각도**: {SEVERITY_LABEL.get(row['severity'], row['severity'])} · **빈도**: {row['frequency']}")
            st.markdown(row["description"] or "_설명 없음_")
        with sc2:
            st.markdown(f"### {row['title_en']}")
            st.caption(f"severity: {row['severity']} · frequency: {row['frequency']}")
            st.markdown(row["description_en"] or "_no description_")

        quotes = row["quotes"] or []
        if quotes:
            st.markdown("**원문 인용**")
            for q in quotes[:5]:
                if isinstance(q, dict):
                    st.markdown(
                        f"> {q.get('text', '')}  \n> <small>review_id: `{q.get('review_id')}`</small>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"> {q}")


# ---------------------------------------------------------------------------
# 탭: 원본 탐색
# ---------------------------------------------------------------------------
def render_explorer_tab(bundle: dict[str, Any]) -> None:
    rows = (bundle.get("clean") or {}).get("rows") or []
    if not rows:
        st.info("정제된 리뷰 데이터가 없습니다.")
        return
    df = pd.DataFrame(rows)

    # cluster 할당 붙이기
    assignments = (bundle.get("clusters") or {}).get("assignments") or {}
    if assignments:
        # review_id 키가 문자열일 수 있음 → 양쪽 대응
        df["cluster_id"] = df["review_id"].astype(str).map(
            {str(k): v for k, v in assignments.items()}
        )

    with st.expander("🔎 필터", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        cluster_opts = sorted([c for c in df.get("cluster_id", pd.Series()).dropna().unique()])
        sel_clusters = f1.multiselect("클러스터", cluster_opts, default=[])
        lang_opts = sorted(df["lang"].dropna().unique().tolist()) if "lang" in df else []
        sel_langs = f2.multiselect("언어", lang_opts, default=[])
        rec_opt = f3.selectbox("recommended", ["전체", "True", "False"], index=0)
        max_pt = int(df["playtime_hours"].max()) if "playtime_hours" in df and len(df) else 0
        pt_range = f4.slider("플레이 시간 (h)", 0, max(max_pt, 1), (0, max(max_pt, 1)))
        query = st.text_input("텍스트 검색 (substring, 대소문자 무시)", "")

    view = df.copy()
    if sel_clusters:
        view = view[view["cluster_id"].isin(sel_clusters)]
    if sel_langs and "lang" in view:
        view = view[view["lang"].isin(sel_langs)]
    if rec_opt != "전체" and "recommended" in view:
        target = rec_opt == "True"
        view = view[view["recommended"].astype(str).str.lower().isin({str(target).lower()})]
    if "playtime_hours" in view:
        view = view[(view["playtime_hours"] >= pt_range[0]) & (view["playtime_hours"] <= pt_range[1])]
    if query.strip() and "text" in view:
        view = view[view["text"].astype(str).str.contains(query, case=False, na=False)]

    total = len(view)
    st.caption(f"필터 결과: **{total:,}** 건")
    page_size = 50
    max_page = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("페이지", min_value=1, max_value=max_page, value=1, step=1)
    start = (page - 1) * page_size
    st.dataframe(view.iloc[start : start + page_size], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 탭: 스팸 후보
# ---------------------------------------------------------------------------
def render_spam_tab(bundle: dict[str, Any]) -> None:
    spam = bundle.get("spam") or {}
    flags = spam.get("flags") or []
    rows = (bundle.get("clean") or {}).get("rows") or []
    if not flags:
        st.info("스팸 후보가 없습니다.")
        return

    text_by_id = {str(r.get("review_id")): r.get("text", "") for r in rows}
    spam_rows = [f for f in flags if f.get("is_spam")]
    stats = spam.get("stats", {})
    pos_thr = stats.get("threshold_pos")
    diff_thr = stats.get("threshold_diff")
    threshold_caption = ""
    if pos_thr is not None:
        threshold_caption = f" · threshold: pos≥{pos_thr:.3f}, diff≥{diff_thr:.3f}"
    st.caption(
        f"탐지된 스팸 후보: **{len(spam_rows):,}** 건{threshold_caption}"
    )

    if "spam_fp" not in st.session_state:
        st.session_state.spam_fp = set()

    df = pd.DataFrame(
        [
            {
                "review_id": f.get("review_id"),
                "score": round(float(f.get("spam_score", 0) or 0), 4),
                "matched_prototype": f.get("matched_prototype", "-"),
                "text": text_by_id.get(str(f.get("review_id")), ""),
            }
            for f in spam_rows
        ]
    )
    if df.empty:
        st.info("스팸으로 판정된 리뷰가 없습니다.")
        return
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    for _, row in df.iterrows():
        rid = str(row["review_id"])
        fp = rid in st.session_state.spam_fp
        header = f"{'✅ (FP 표시됨)' if fp else '🚨'} `{rid}` · score={row['score']:.3f} · {row['matched_prototype']}"
        with st.expander(header, expanded=False):
            st.write(row["text"] or "_(본문 없음)_")
            if st.toggle("False positive 로 표시 (세션 한정)", value=fp, key=f"fp_{rid}"):
                st.session_state.spam_fp.add(rid)
            else:
                st.session_state.spam_fp.discard(rid)

    st.caption(f"이번 세션 FP 표시: {len(st.session_state.spam_fp)} 건 (저장되지 않음)")


# ---------------------------------------------------------------------------
# 탭: 다운로드
# ---------------------------------------------------------------------------
def render_download_tab(bundle: dict[str, Any]) -> None:
    sha1 = bundle.get("sha1", "unknown")
    report = bundle.get("report_md") or ""
    jira = bundle.get("jira")
    cost = bundle.get("cost")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📄 report.md",
            data=report or "리포트가 아직 생성되지 않았습니다.",
            file_name=f"report_{sha1}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not report,
        )
    with c2:
        st.download_button(
            "🧾 jira.json",
            data=json.dumps(jira, ensure_ascii=False, indent=2) if jira else "{}",
            file_name=f"jira_{sha1}.json",
            mime="application/json",
            use_container_width=True,
            disabled=jira is None,
        )
    with c3:
        st.download_button(
            "💰 cost.json",
            data=json.dumps(cost, ensure_ascii=False, indent=2) if cost else "{}",
            file_name=f"cost_{sha1}.json",
            mime="application/json",
            use_container_width=True,
            disabled=cost is None,
        )

    st.caption(f"artifact 위치: `{bundle.get('dir', '-')}`")
    with st.expander("report.md 미리보기", expanded=False):
        st.markdown(report or "_비어있음_")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Steam 리뷰 분류·요약",
        page_icon="🎮",
        layout="wide",
    )
    st.title("🎮 Steam 리뷰 분류·요약 대시보드")
    st.caption(
        "CX팀 매니저용 — 5,000건 Steam 리뷰를 주제별로 묶고 부정 리뷰 이유를 추출해 개발팀에 전달합니다."
    )

    # 첫 진입 시 기본 CSV 의 precomputed artifact 가 있으면 자동 로드
    if "bundle" not in st.session_state:
        st.session_state.bundle = None
        if DATA_DEFAULT.exists():
            try:
                default_sha = sha1_of_file(DATA_DEFAULT)
                cached = load_artifacts(default_sha)
                if cached:
                    st.session_state.bundle = cached
            except Exception:
                pass

    input_path, start = render_sidebar(st.session_state.bundle)

    if start:
        if input_path is None:
            st.error("입력 CSV 를 먼저 선택해주세요.")
        elif not _get_api_key() and PIPELINE_AVAILABLE:
            st.error(
                "OPENAI_API_KEY 가 설정되지 않아 분석을 시작할 수 없습니다. "
                "`.env` 또는 Streamlit Cloud Secrets 에 키를 추가해주세요."
            )
        else:
            ok, msg = validate_csv(input_path)
            if not ok:
                st.error(msg)
            else:
                result = run_with_progress(input_path)
                if result is not None:
                    # sha1 / dir 정보 주입 (pipeline 이 안 채워줬을 경우 대비)
                    result.setdefault("sha1", sha1_of_file(input_path))
                    result.setdefault("dir", str(artifacts_dir(result["sha1"])))
                    st.session_state.bundle = result
                    st.session_state.last_run_at = datetime.now().isoformat(timespec="seconds")

    bundle = st.session_state.bundle
    if not bundle:
        st.info(
            "👈 왼쪽 사이드바에서 입력 CSV 를 선택하고 **분석 시작** 버튼을 눌러주세요.\n\n"
            "기본 `data/reviews.csv` 의 precomputed 결과가 저장돼 있으면 자동으로 표시됩니다."
        )
        if not PIPELINE_AVAILABLE:
            st.warning("현재 백엔드가 준비 중입니다 (데모 모드).")
        return

    render_kpis(bundle)
    st.divider()

    tabs = st.tabs(
        ["📑 주제별 보기", "🚨 부정 이유 보기", "🔍 원본 탐색", "🛡 스팸 후보", "📥 다운로드"]
    )
    with tabs[0]:
        render_topics_tab(bundle)
    with tabs[1]:
        render_negatives_tab(bundle)
    with tabs[2]:
        render_explorer_tab(bundle)
    with tabs[3]:
        render_spam_tab(bundle)
    with tabs[4]:
        render_download_tab(bundle)


if __name__ == "__main__":
    main()
