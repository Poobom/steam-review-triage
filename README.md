# Steam 리뷰 분류·요약 파이프라인

> Steam 리뷰 5,000건을 주제별로 자동 분류하고, 부정 리뷰의 원인을 개발팀에 전달 가능한 형태(Markdown / Jira JSON)로 뽑아주는 브라우저 기반 도구.

---

## 무엇을 해주나

CX팀 김지원 매니저님(비개발자·크롬 브라우저만 사용)을 위해 만들어졌습니다.

- 주간 리뷰 5,000건을 **수동 분류 이틀 → 3~5분**으로 단축합니다.
- LLM 기반 **의미 임베딩 + 클러스터링**으로 "graphics are stunning" 과 "그래픽이 정말 아름답다" 를 같은 주제로 묶습니다.
- 부정 리뷰만 따로 모아 **원인 + 심각도(critical/major/minor) + 대표 인용**을 정리합니다.
- 결과를 **Markdown 리포트** 또는 **Jira 이슈 JSON(bulk import 호환)** 로 한 번에 내보냅니다.
- 한국어·영어 병기 — 글로벌 개발팀과 바로 공유 가능합니다.

---

## 배포 URL

**[배포 후 업데이트]**

> Streamlit Community Cloud 에 public repo 로 배포됩니다. 첫 방문 시 `reviews.csv` 기본 분석 결과가 즉시 표시되며(precomputed artifacts), 분석 시작 버튼을 누르면 캐시 hit 으로 약 5초 만에 6단계 파이프라인이 replay 됩니다.

---

## 빠른 시작 (사용자 가이드)

비개발자 김 매니저님용 4단계입니다. Python 이나 터미널이 필요 없습니다.

1. 위 **배포 URL** 을 크롬에서 엽니다.
2. 상단의 KPI 카드(총 리뷰 수, 스팸 수, 클러스터 수, 비용)와 이미 precompute 된 분석 결과를 바로 확인합니다.
3. 새 데이터를 분석하려면 사이드바의 **"다른 CSV 업로드"** 에서 파일을 올리고 **"분석 시작"** 버튼을 누릅니다.
   - 진행률 바와 단계별 로그가 표시됩니다 (약 3분).
   - 동일 CSV 를 다시 누르면 캐시 hit 으로 5초 replay 됩니다.
4. **📥 다운로드** 탭에서 `report.md` 와 `jira.json` 을 내려받아 개발팀에 전달합니다.

> 탭 구성: **📑 주제별 / 🚨 부정 이유 / 🔍 원본 탐색 / 🛡 스팸 후보 / 📥 다운로드**

---

## 로컬 개발 환경 셋업

```bash
# 1. clone
git clone https://github.com/<user>/<repo>.git
cd <repo>

# 2. 가상환경 + 의존성
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. .env 설정
cp .env.example .env
# .env 파일을 열어 키를 채웁니다 (OpenAI 한 개만 필요):
#   OPENAI_API_KEY=sk-...        (임베딩 + LLM 요약 공용)

# 4. CLI 파이프라인 1회 실행 (artifacts 생성 + 비용 로그)
python pipeline.py --input data/reviews.csv

# 5. Streamlit 앱 실행
streamlit run app.py
# → http://localhost:8501
```

필수 파이썬: 3.10+. `requirements.txt` 에는 `streamlit`, `openai`, `pandas`, `numpy`, `scikit-learn`, `python-dotenv` 가 포함됩니다. LLM(요약·부정 이유)과 임베딩 모두 OpenAI 한 벤더로 통일했으므로 별도 Anthropic SDK 는 필요하지 않습니다.

---

## 파이프라인 6단계 개요

`pipeline.py` 가 호출하고, `src/` 의 모듈들이 단계별 산출물을 `artifacts/{sha1}/` 에 저장합니다. SHA1 은 입력 CSV 의 해시 → 동일 입력은 캐시 hit.

| # | 단계 | 설명 | LLM 비용 |
|---|---|---|---|
| 1 | **Load & Clean** | CSV 로드, 인코딩 깨진 3% latin-1→utf-8 휴리스틱 복구 | $0 |
| 2 | **Spam Detection** | 양/음 프로토타입 임베딩 + URL 보조 신호로 zero-shot 탐지 | 단계 3 과 공유 |
| 3 | **Clustering** | PCA 50 차원 → KMeans (k 자동 선택, 6~10 범위 silhouette 기반) + MMR 다양성 샘플링 | ~$0.02 (임베딩 전체) |
| 4 | **Cluster Naming + Summary** | 클러스터당 대표 10건을 OpenAI gpt-4o-mini 에 넘겨 주제명·요약·키워드(한/영 병기) 생성 | 포함 |
| 5 | **Negative Reasons** | `recommended=False` 만 필터, 불만 이유 + 심각도 + 대표 인용 추출 (gpt-4o-mini) | 포함 |
| 6 | **Export** | `report.md`, `jira.json`, `cost.json` 생성 (LLM 호출 없음) | $0 |

**실측 비용: $0.0127 / 1 회 (5,000건)** — 예상 $0.22 의 약 5% 수준이며, $3 한도 대비 0.5% 미만. 주간 5,000건 분석을 반복해도 비용 부담이 사실상 없습니다.

---

## 비용

1 회 전체 분석 **실측 비용: $0.0127** (5,000건, 한/영 병기 포함). 초기 플랜 예상치($0.22)의 **약 5%** 수준으로, $3 한도 대비 0.5% 미만입니다.

| 모델 | 용도 | 단가 | 이번 실행 실측 |
|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 임베딩 (5,000건) | $0.020 / 1M tokens | ~$0.002 |
| `gpt-4o-mini` (OpenAI) | 클러스터 요약 + 부정 이유 | input $0.150 / 1M, output $0.600 / 1M | ~$0.011 |

플랜 단계에서는 Anthropic Claude Haiku 4.5 를 고려했으나, 실제 구현에서는 **OpenAI gpt-4o-mini 한 모델로 통일**하여 벤더·SDK·키 관리를 단순화했습니다. 누적 비용은 `artifacts/{sha1}/cost.json` 과 사이드바 **누적 비용** 표시에서 확인할 수 있습니다. 동일 CSV 재분석은 캐시 hit 으로 **$0**.

---

## 출력 포맷

두 포맷 모두 **📥 다운로드** 탭에서 내려받을 수 있습니다.

### 1. Markdown 리포트 (`report.md`)
- KPI 요약
- 주제별 카드 (주제명 / 요약 3 줄 / 대표 인용 / 키워드)
- 부정 이유 표 (심각도 + 빈도 + 인용)
- 한국어 + 영어 병기

### 2. Jira 이슈 JSON (`jira.json`)
- Jira REST API `POST /rest/api/3/issue/bulk` 호환 스키마
- `severity` → `priority` 매핑: `critical`→Highest, `major`→High, `minor`→Medium
- 기본 `project.key` 는 **"GAME"** — 환경별로 바꾸려면 JSON 파일의 `fields.project.key` 를 일괄 치환하거나, `src/export.py` 의 `JIRA_PROJECT_KEY` 상수를 수정하고 파이프라인을 재실행하세요.
- ADF 풀스펙 대신 plain text `description` 사용 (검수 용이).
- 부정 클러스터만 이슈로 변환되며, 긍정/mixed 는 Markdown 에만 포함됩니다.

---

## 데이터 출처

- **Kaggle 데이터셋**: [`najzeko/steam-reviews-2021`](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021)
- **샘플링**: 위처 3(The Witcher 3) 리뷰 중 5,000건 샘플
- **컬럼**: `review_id`, `text`, `lang`, `recommended`, `playtime_hours`, `posted_at`
- **언어 분포**: english 60%, koreana 25%, 기타 15%
- **의도된 노이즈**: 스팸·광고 5% (5 종 고정 문구 기반 치환), 인코딩 깨짐 3% (utf-8 → latin-1 재인코딩)

원본 생성 로직은 과제 패키지의 `prepare_reviews.py` 를 참고하세요.

---

## 보안 / 한계

- **API 키 관리**: 로컬은 `.env` (gitignore 됨), Streamlit Cloud 는 Secrets UI 에 `OPENAI_API_KEY` **한 개만** 등록하면 됩니다(임베딩·LLM 공용). 키는 repo 에 절대 commit 하지 않습니다.
- **Streamlit Cloud 콜드 스타트**: 7 일 무사용 sleep 후 첫 접속은 ~30 초 지연이 있을 수 있습니다. 데모 영상 촬영 전 미리 한 번 접속해 warm up 하세요.
- **인증 없음**: URL 을 아는 누구나 접근 가능합니다. 실제 운영 시에는 Streamlit Community Cloud 의 viewer 제한 또는 별도 auth gateway 가 필요합니다.
- **스팸 탐지 정확도 미보증**: LLM 검증 단계를 생략했습니다 (시간 제약). 세션 내 false positive 토글로 수동 보정 가능.
- **데이터 프라이버시**: 업로드한 CSV 는 Streamlit Cloud 의 런타임 파일시스템에만 저장되며, 영구 저장소로는 전송되지 않습니다. 단 OpenAI API 로 전송되는 텍스트는 OpenAI 의 정책을 따릅니다.

---

## 디렉토리 구조

```
task01_solution/
├── README.md                   # (이 파일)
├── tradeoffs.md                # 포기 기능과 사유
├── progress.md                 # 30분 단위 진행 로그
├── .gitignore                  # *.csv 에 !data/reviews.csv 예외 추가
├── .env.example                # 키 템플릿 (.env 는 gitignore)
├── .streamlit/
│   └── config.toml             # Streamlit 테마/서버 설정
├── requirements.txt
├── pipeline.py                 # CLI 진입점
├── app.py                      # Streamlit 진입점
├── src/
│   ├── clean.py                # 단계 1: 로드 + 인코딩 복구
│   ├── embed.py                # 임베딩 공통
│   ├── spam.py                 # 단계 2: 스팸 탐지
│   ├── cluster.py              # 단계 3: KMeans + MMR
│   ├── topics.py               # 단계 4: 주제·요약
│   ├── negatives.py            # 단계 5: 부정 이유
│   ├── export.py               # 단계 6: Markdown + Jira JSON
│   ├── llm.py                  # LLM 래퍼 + 비용 집계
│   └── cache.py                # SHA1 디스크 캐시
├── tests/                      # 단위·통합 테스트
├── artifacts/
│   └── {sha1}/                 # 입력 해시별 산출물 폴더
│       ├── 01_clean.json
│       ├── 02_spam.json
│       ├── 03_clusters.json
│       ├── 04_topics.json
│       ├── 05_negatives.json
│       ├── 06_report.md
│       ├── 06_jira.json
│       └── cost.json
└── data/
    └── reviews.csv             # 데모용 기본 데이터 (gitignore 예외)
```

---

## 기여 / 문의

- 본 저장소는 Krafton AI FDE 모의 과제 1 제출물이며, 3 시간 제약 하에 구현되었습니다.
- 의도적으로 포기한 기능 목록은 [`tradeoffs.md`](./tradeoffs.md) 를 참고하세요.
- 진행 과정 타임라인은 [`progress.md`](./progress.md) 에 30 분 단위로 기록되어 있습니다.
- 버그 / 개선 제안은 GitHub Issue 로 남겨주세요.
