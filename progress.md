# Progress Log

## 17:26 - 시작
- 데이터셋 najzeko/steam-reviews-2021 다운로드 중

## 17:36~ - 데이터 탐색 & 플랜 모드 의사결정

### 입력 데이터 확인
- `task01_data/reviews.csv` (5,000건, ~1MB) — review_id, text, lang, recommended, playtime_hours, posted_at
- `task01_data/prepare_reviews.py` — 원본을 `t.encode("utf-8").decode("latin-1")` 로 일부러 깨뜨림 → 역연산 `s.encode("latin-1").decode("utf-8", errors="ignore")` 로 복구 가능
- `task01_data/steam_reviews.csv` (8.1GB) — 원본, 사용 X
- 데이터 특성: 영어 60% / 한국어 25% / 기타 15%, 위처3 리뷰 다수, 스팸 ~5%, 인코딩 깨짐 ~3%

### 인프라 결정
| 항목 | 결정 | 근거 |
|---|---|---|
| 임베딩 공급자 | **OpenAI `text-embedding-3-small`** | 5,000건 ≈ $0.02, 다국어(한/영/기타) 우수. 신규 키 발급 |
| 배포 플랫폼 | **Streamlit Community Cloud** | 비개발자 사용자(김 매니저) URL 클릭 한 번. 개발자 입장에서도 가장 빠른 배포 |
| Repo 전략 | task01_solution에서 새 public GitHub repo 생성/푸시 | 제출물 #2 요건 |
| 진행률 UX | **옵션 B + 파일 hash 캐시** (단순화) | 시스템답게 동작 (다른 CSV도 처리). 같은 파일 두번째 클릭은 캐시 즉시 반환 |

### 진행률 UX 옵션 비교 결과
- 옵션 A (precompute + replay): 추가 비용 0이지만 "정적 데모 페이지"라 시스템 본질 X → 기각
- 옵션 B (실제 LLM 매번 실행): 1회 ~$0.25, $3 한도 = 12회 가능. 충분히 안전 → **채택**
- 옵션 C (하이브리드): 분기 복잡, 시간 부족 → 기각
- 옵션 D (캐시 + 강제 재실행 버튼): 정직성↑이지만 과한 복잡도 → 기각
- 비용 해석: "$3 이하"는 채점 시 누적 사용량으로 해석. 1회 ~$0.25면 안전권

### 파이프라인 6단계 설계 (단계별 점검 완료)

#### 단계 1 — Load & Clean (LLM 호출 0)
- pandas로 로드 → 인코딩 깨진 3% latin-1 역연산 복구
- 두 버전(원본/복구) 비교해 더 정상적인 쪽 선택 (false positive 방지)
- 길이 필터 제거 → 짧은 정상 리뷰("ok", "최고") 살리기. 클러스터링에서 자연스럽게 흡수

#### 단계 2 — Spam Detection (Bipolar Zero-shot, 임베딩만, $0.02)
- prepare_reviews.py 5종 문구 그대로 X → 의역에도 강한 "유형 설명" 프로토타입 사용
- 양 프로토타입 5개 (광고/현금거래/홍보/스폰서/봇)
- 음 프로토타입 3개 (정상 리뷰 신호)
- 판정: `max(sim_양) - max(sim_음) > 0.05 AND max(sim_양) > 0.4`
- threshold는 데이터 분포 히스토그램 기반 자동 튜닝 (5% 분위수)
- URL 정규식은 보조 점수(+0.1)로만 — "구조적 마커"로 README에 명시 (키워드 매칭과 구분)
- LLM 검증 생략 (시간/비용 절약)
- **별도 "스팸 후보" 탭 제공** → 김 매니저 spot check 가능

#### 단계 3 — Clustering (LLM 호출 0)
- 단계 2 임베딩 재사용 (스팸 제외, ~4,750건)
- **KMeans + silhouette로 k=8~16 자동 선택** (k=12는 가설값일 뿐)
- PCA 50차원 축소 (속도 + 노이즈 제거)
- 작은 클러스터(<20건) → "기타" 병합
- 대표 샘플: nearest 5 + MMR 다양성 5 = 클러스터당 10개 (다양성+대표성)
- 게임 리뷰 예상 12개 주제 추정: 그래픽/스토리/게임플레이/버그/가격/콘텐츠분량/사운드/UI/DLC/짧은추천/비교/난이도

#### 단계 4 — Cluster Naming + Summary (Haiku 4.5, 12~15 call, ~$0.07)
- 모델: claude-haiku-4-5-20251001 (Sonnet은 5배 비싸고 분류엔 과잉)
- **한국어 + 영어 동시 출력** (글로벌 개발팀 대응, 비용 +$0.02만 증가)
- 출력 스키마: `topic`, `topic_en`, `summary[3]`, `summary_en[3]`, `sentiment`, `keywords_ko`, `keywords_en`, `representative_quotes[]` (review_id + 원문)
- JSON mode 미지원 → "JSON만 출력" 지시 + 파싱 실패 시 1회 재시도

#### 단계 5 — Negative Reasons (Haiku, 8~12 call, ~$0.13)
- recommended=False만 필터 (~1,400건 추정)
- mixed sentiment 추출은 MVP에서 미구현 → tradeoffs.md에 명시
- 클러스터별로 nearest 10 + MMR 10 = 20건 샘플링 후 LLM 호출
- 출력: `issues[]` 각각 `title/title_en/description/description_en/severity(critical|major|minor)/frequency/representative_quotes`
- "구체적 재현 조건/대상 명시" 지시 + "원문 명시 정보만" (환각 방지)

#### 단계 6 — Export (LLM 호출 0)
- **Markdown 리포트** (`06_report.md`): KPI 요약 → 클러스터별 주제 분석 → 부정 이슈 표
- **Jira JSON** (`06_jira.json`): bulk create 호환. ADF 풀스펙 대신 plain text description (검수 쉬움)
- severity → priority 매핑 (critical→Highest 등)
- 부정 클러스터 이슈만 Jira에 포함, 긍정/mixed는 Markdown 리포트에만
- `cost.json` 별도 분리 (제출물 "사용량 로그 필수" 충족)

### 누적 비용 추산
| 단계 | 비용 |
|---|---|
| 임베딩 (단계 2 공유) | $0.02 |
| Cluster naming (단계 4) | $0.07 |
| Negative reasons (단계 5) | $0.13 |
| **합계** | **~$0.22** ($3 한도 대비 매우 안전) |

### 금지사항 해석 정리
- "리뷰 5,000건 통째로 LLM 프롬프트에 붙이기 금지" = **API 호출 prompt** 안에 5,000건 concat 금지. Claude Code가 디버깅 차원에서 CSV Read하는 것은 무관.
- "단순 키워드 매칭 금지" = 임베딩 cosine + URL 구조 신호로 회피. 의역 10건 테스트 통과 가능.

### 다음 30분 계획
- Plan 모드 종료 → 구현 시작
- 단계 1~3 (오프라인 파이프라인 코드) 작성
- artifacts 디렉토리 구조 확정 (`artifacts/{file_hash}/01_clean.json` ...)

## 18:30~19:10 — 구현 1차 (코드 작성)

### Scaffold
- `requirements.txt`, `src/`, `.env` (.gitignore 됨), `.env.example`
- `.gitignore` 에 `!data/reviews.csv` 예외 추가 (데이터 포함 필수)
- `.streamlit/config.toml`, `data/reviews.csv` 복사

### 핵심 모듈 작성
- `src/clean.py` — 인코딩 휴리스틱 (mojibake marker + 한글/영문 비율로 더 정상적인 쪽 선택)
- `src/llm.py` — OpenAI 클라이언트 + CostTracker (단계별 토큰/비용 집계)
- `src/embed.py` — text-embedding-3-small 배치 (256개씩) + cosine_similarity_matrix 헬퍼
- `src/spam.py` — 양 5/음 3 프로토타입 zero-shot + URL 정규식 보조 (구조적 마커)
- `src/cluster.py` — PCA 50차원 → KMeans (silhouette 자동 k 선택, 1500 샘플) + MMR 다양성 샘플링
- `src/topics.py` — gpt-4o-mini JSON mode, 한/영 병기, representative_quotes
- `src/negatives.py` — recommended=False 만, 클러스터당 nearest 10 + MMR 10 = 20건 샘플링
- `src/export.py` — Markdown KPI/주제/이슈 표 + Jira bulk-create JSON (severity→priority 매핑)
- `src/cache.py` — SHA1 12자 prefix 폴더 캐시
- `pipeline.py` — 6단계 통합, on_step 콜백, --force 옵션

### 모델 변경
- 플랜의 Anthropic Haiku 4.5 → **OpenAI gpt-4o-mini** 로 통일 (사용자 키 1개로 단순화)

### 병렬 작업 (sub-agent)
- sub A: `app.py` (Streamlit UI 450줄) — 작성 완료
- sub B: `README.md` + `tradeoffs.md` — 작성 완료
- 메인: `src/*` + `pipeline.py`

### 사용자 외부 작업
- GitHub repo 생성, OpenAI Hard limit 설정, Streamlit Cloud 가입, 데모 영상 도구 준비

## 19:10 — 파이프라인 1회 실행 검증

### 실행 명령
```
python pipeline.py --input data/reviews.csv
```

### 결과
- **단계 1 (정제)**: 5,000 → 4,982건, 인코딩 복구 60건 (예상 ~150건보다 적음 — 휴리스틱이 보수적, 일부 깨진 텍스트는 utf-8 으로 떨어지지 않아 보존)
- **단계 2 (임베딩+스팸)**: 임베딩 4,982건 ~38초 / 스팸 162건 (3.25%, 예상 5%보다 적음 — 양/음 차이 임계값이 보수적)
- **단계 3 (클러스터링)**: k=16 (silhouette 0.1294), other=0
- **단계 4 (주제 명명)**: 16개 클러스터 모두 명명 완료 (~73초)
- **단계 5 (부정 이유)**: 16개 클러스터 처리, 이슈 총 29개 추출 (~84초)
- **단계 6 (export)**: Markdown + Jira JSON 생성
- **총 비용: $0.0127** (예상 $0.22의 6%, gpt-4o-mini 가 매우 저렴)
- **총 시간**: 약 3분 20초
- 출력 폴더: `artifacts/aba8cd89ec4e/`

### 사소한 issue
- `spam.py` 의 URL_PATTERN 에 capturing group 경고 → non-capturing `(?:...)` 로 정리 필요 (기능엔 영향 없음)
- Windows 콘솔 cp949 인코딩 문제로 한글 진행 로그가 깨져 보임 — 데이터 자체는 정상

### 다음 30분 계획 (19:10~19:40)
- 의역 테스트: "graphics are stunning" / "visuals are breathtaking" / "그래픽이 정말 아름답다" 가 같은 클러스터에 들어가는지 검증
- 스팸 의역 테스트: 원본 5종 문구의 의역("체이크한 CS 스킨 싸게 팝니다") 탐지 여부 검증
- `README.md` 의 모델명 수정 (Haiku → gpt-4o-mini)
- `app.py` 로컬 streamlit 실행 확인
- 산출물 spot check: 주제 16개 명명 품질, 부정 이슈 29개 actionable 한지
