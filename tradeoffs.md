# Tradeoffs — 3시간 제한 속에서 의도적으로 포기한 기능

본 프로젝트는 Krafton AI FDE 모의 과제 1 의 제약인 **3시간 구현 + LLM 비용 $3 이하 + Streamlit Cloud 1-URL 배포** 를 만족시키기 위해, "반드시 동작하는 end-to-end 파이프라인" 을 우선순위로 삼았습니다. 그 결과 아래 8 가지 기능은 **기술적으로 난이도가 높아서가 아니라**, 3시간 안에 넣으면 핵심 경로(6단계 파이프라인 + 진행률 UI + 배포)가 불안정해질 위험이 크기 때문에 의도적으로 배제했습니다. 각 항목마다 현재 우회 방법과 추후 추가 시 예상 작업량을 함께 기록합니다.

> **플랜 대비 주요 변경 — LLM 단일화**: 초기 플랜에서는 요약/부정 이유 추출에 Anthropic Claude Haiku 4.5 를, 임베딩에 OpenAI 를 썼습니다. 실제 구현에서는 **OpenAI `gpt-4o-mini` 한 모델로 통일**해 벤더·SDK·키 관리를 단순화했습니다. 그 결과 실측 1 회 실행 비용은 **$0.0127** (예상 $0.22 의 약 5%) 로 내려갔으며, 5,000 건 규모의 주간 분석을 반복해도 $3 한도의 0.5% 미만만 소모합니다. 클러스터 수는 silhouette 기반 **자동 선택 (6~10 범위)** 으로 조정했습니다 — 현재 실측은 k=16 (silhouette 0.13) 이나 후속 fix 에서 6~10 범위로 수렴시킬 예정입니다.

---

### 1. 자동 데이터 fetching (매주 Steam API)

**현재 상태**: 미구현

**왜 포기했는가**: Steam Web API 인증·레이트리밋·증분 수집·스케줄러(Cloud Scheduler 또는 GitHub Actions cron) 구성을 제대로 하려면 최소 45~60 분이 필요하고, Streamlit Community Cloud 는 백그라운드 워커를 지원하지 않아 별도 인프라(예: GitHub Actions + S3)가 필요합니다. 3 시간 예산 안에서 핵심 파이프라인 정확도 검증(의역 테스트 등)을 희생하게 됩니다.

**대안/회피 방법**: 사이드바의 **"다른 CSV 업로드"** 컴포넌트로 매니저가 매주 한 번 새 CSV 를 수동 업로드하는 운영 모델을 채택. 업로드한 CSV 는 SHA1 해시로 캐싱되므로 두 번째 클릭부터는 비용 $0.

**나중에 추가하려면**: 약 2~3 시간 작업. ① Steam API 크롤러 모듈(`src/fetch.py`) — appid + 언어별 페이지네이션 + 증분 cursor 저장, ② GitHub Actions cron(`.github/workflows/weekly.yml`) 으로 매주 월요일 실행 후 `data/reviews.csv` 갱신 commit, ③ `pipeline.py --auto` 플래그로 CI 내부에서 artifacts 재생성 후 같은 commit 에 포함.

---

### 2. Mixed sentiment 추출 (recommended=True 인데 부정적 텍스트)

**현재 상태**: 미구현

**왜 포기했는가**: 단계 5 의 부정 이유 추출은 `recommended=False` 만 필터링합니다. `recommended=True` 인데 본문은 비판적(예: "재밌긴 한데 서버 렉 너무 심함") 인 **mixed** 케이스는 전체의 10~15% 로 추정되며, 이를 다루려면 단계 4 의 sentiment 필드를 신뢰 가능하게 만들고, 단계 5 를 "recommended 무시, sentiment 기준으로 재분류" 로 리팩토링해야 합니다. 추가 LLM 호출(~$0.08)과 검증 루프가 필요합니다.

**대안/회피 방법**: 단계 4 프롬프트에 `sentiment: "positive | negative | mixed"` 필드를 이미 포함시켜 **주제별 보기 탭에서는 mixed 클러스터가 식별**됩니다. 단 개별 리뷰 단위의 mixed 탐지는 되지 않습니다.

**나중에 추가하려면**: 약 60 분. 단계 5 의 input filter 를 `sentiment in ['negative', 'mixed']` 로 확장하고, 리뷰 단위 sentiment 분류기(클러스터 평균이 아닌 per-review) 를 gpt-4o-mini 에 batch 호출로 추가. gpt-4o-mini 단가(input $0.150 / 1M, output $0.600 / 1M) 기준 추가 비용 ~$0.005 수준.

---

### 3. 다중 게임 동시 분석 (게임별 분리 dashboard)

**현재 상태**: 미구현 — 단일 게임(위처 3) 단일 CSV 가정

**왜 포기했는가**: 과제 지급 데이터가 단일 게임 5,000 건이고, 다중 게임을 지원하려면 `game_id` 컬럼 추가 → 파이프라인 전반에 per-game 분기 → UI 에 게임 선택 드롭다운 → artifacts 구조를 `artifacts/{game_id}/{sha1}/` 로 중첩해야 합니다. 1 시간 이상의 리팩토링이 필요하고, 현재 과제 평가 기준에서 얻는 점수가 불확실합니다.

**대안/회피 방법**: 게임별로 별도 CSV 를 만들어 사이드바에서 각각 업로드하면 SHA1 캐시가 자동 분리되어 사실상 게임별 독립 분석이 가능합니다. 단 한 화면에서 비교는 불가.

**나중에 추가하려면**: 약 3~4 시간. ① CSV 스키마에 `game_id` 추가, ② `pipeline.py` 에 `--game-id` 옵션 + per-game artifacts 폴더, ③ Streamlit 메인 페이지 상단에 게임 선택 드롭다운 + 비교 모드 탭, ④ 게임 간 주제 매칭(공통 버그인지 게임 고유 이슈인지)은 별도 임베딩 유사도 단계 필요.

---

### 4. 로그인 / 멀티 테넌시·인증

**현재 상태**: 미구현 — URL 을 아는 누구나 접근 가능

**왜 포기했는가**: 과제 채점 시나리오는 "매니저 한 명이 URL 로 접근" 이며, 외부 공격 시나리오는 가정에 포함되지 않습니다. 인증을 붙이려면 OAuth(Google/GitHub) 프로바이더 연동 또는 `streamlit-authenticator` 추가 + 세션 스토리지 설계가 필요하고, Streamlit Community Cloud 무료 티어는 private app 이 제한적입니다.

**대안/회피 방법**: 업로드된 CSV 는 artifacts 폴더에 평문 저장되지만, GitHub repo 에는 commit 되지 않고 Streamlit Cloud 재배포 시 초기화됩니다. URL 자체를 공유 대상에게만 배포하는 "link-only" 보안 모델을 채택.

**나중에 추가하려면**: 약 2 시간. ① `streamlit-authenticator` 또는 Google OAuth 연동, ② 사용자별 artifacts 폴더 격리(`artifacts/{user_id}/{sha1}/`), ③ 감사 로그. 실운영 환경이라면 Streamlit 대신 FastAPI + Next.js 분리 스택을 검토.

---

### 5. 모델 성능 평가 지표 수치화 (precision/recall + ground truth 라벨)

**현재 상태**: 미구현 — spot check 만 진행

**왜 포기했는가**: 채점 시 투입되는 의역 10 건 테스트는 검증하지만, 전체 5,000 건에 대한 정량적 precision/recall 은 **ground truth 라벨이 존재하지 않기 때문에** 측정 불가능합니다. 라벨을 직접 만들려면 최소 500 건을 수동 분류해야 하고(2~3 시간), LLM 을 judge 로 쓰는 방식은 self-consistency 함정이 있습니다.

**대안/회피 방법**: ① 과제 Verification 체크리스트의 의역 3 종 + 스팸 의역 1 종 테스트로 spot check, ② silhouette score 와 클러스터 크기 분포를 `cost.json` 에 기록해 클러스터 품질 근사치 확인, ③ 대표 인용 샘플을 리포트에 노출해 매니저가 직관적으로 검수 가능하게 함.

**나중에 추가하려면**: 약 4~6 시간. ① 500 건 golden set 수동 라벨링 (또는 GPT-4o 를 judge 로 두되 human-in-the-loop 재검토), ② `eval.py` 스크립트로 클러스터 vs 라벨 matching(Hungarian algorithm) + classification report, ③ Streamlit 에 모델 평가 탭 추가.

---

### 6. 모바일 UI 최적화

**현재 상태**: 미구현 — 데스크톱 크롬 기준만 검증

**왜 포기했는가**: 실사용자 김지원 매니저는 "크롬 브라우저만 사용" 이 명시되어 있고, 데이터 표/대시보드 특성상 좁은 화면에서는 정보 밀도가 급격히 떨어집니다. Streamlit 의 반응형 breakpoint 를 직접 제어하려면 커스텀 CSS injection 이 필요하고, 5 개 탭 각각을 모바일 레이아웃으로 재설계하는 데 1 시간 이상 소요됩니다.

**대안/회피 방법**: `.streamlit/config.toml` 의 테마 설정만 적용하고, 모바일 접근 시에는 Streamlit 기본 반응형에 의존. 중요한 다운로드 버튼은 상단 배치로 모바일에서도 접근 가능.

**나중에 추가하려면**: 약 2~3 시간. ① 커스텀 CSS 로 좁은 화면 breakpoint 에서 탭 → 세로 아코디언 변환, ② KPI 카드 2x2 → 1 열 스택, ③ DataFrame 을 카드 리스트로 대체하는 래퍼 함수.

---

### 7. Streamlit Cloud 콜드 스타트 ~30초 (7일 무사용 sleep)

**현재 상태**: 수용함 — 해결 방법 미구현

**왜 포기했는가**: Streamlit Community Cloud 무료 티어의 정책상 7 일 무사용 앱은 자동 sleep 되며, 첫 접속 시 ~30 초 콜드 스타트가 발생합니다. 이를 회피하려면 유료 플랜 전환 또는 외부 cron 으로 주기적 ping 을 때려야 하는데, 유료 플랜은 과제 범위를 벗어나고 ping 트릭은 약관 위반 소지가 있습니다.

**대안/회피 방법**: ① 데모 영상 촬영 10 분 전 미리 URL 에 한 번 접속해 warm up, ② README 의 **보안/한계** 섹션에 명시해 채점관 기대치 조정, ③ 첫 방문자가 바로 의미 있는 결과를 보도록 precomputed artifacts 를 repo 에 commit → 30 초 후엔 캐시 hit 으로 즉시 응답.

**나중에 추가하려면**: 2 가지 선택지. ① Streamlit Cloud 유료 플랜(Creator $20/월) 으로 always-on 활성화, ② Render / Fly.io / Railway 로 이관(월 $5~10, 콜드 스타트 짧음), ③ GitHub Actions 로 30 분마다 HEAD 요청(임시 방편, 권장 X).

---

### 8. 실시간 알림 / Slack 연동 (critical 이슈 자동 통지)

**현재 상태**: 미구현

**왜 포기했는가**: 단계 5 에서 `severity=critical` 이슈(게임 진행 차단급 버그) 가 발견되면 Slack / Discord / 이메일로 즉시 알림을 보내는 기능은 유용하지만, ① webhook URL 을 secrets 로 관리, ② severity threshold 를 UI 에서 설정 가능하게, ③ 중복 알림 방지(기존 artifacts 와 diff) 로직까지 합치면 1 시간 이상 소요됩니다. 또한 **비동기 알림은 Streamlit 런타임이 종료된 후 발생해야 진정한 가치**가 있는데, Streamlit Cloud 는 백그라운드 프로세스를 지원하지 않습니다.

**대안/회피 방법**: Jira JSON 다운로드 기능이 사실상 이 역할의 동기 버전을 수행합니다. 매니저가 분석 직후 Jira 로 bulk import 하면 개발팀의 기존 알림 채널(Jira → Slack 연동)로 자동 전파됩니다.

**나중에 추가하려면**: 약 1.5~2 시간. ① `src/notify.py` 에 Slack Incoming Webhook 호출 모듈, ② `pipeline.py --notify` 플래그 + `.env` 에 `SLACK_WEBHOOK_URL`, ③ 이전 실행의 `05_negatives.json` 과 diff 해서 신규 critical 만 전송, ④ GitHub Actions 주간 실행(1 번 항목)과 연계하면 완전 자동화 가능.

---

## 요약

| # | 기능 | 대안 | 추가 작업량 |
|---|---|---|---|
| 1 | 자동 fetching | 수동 CSV 업로드 | 2~3h |
| 2 | Mixed sentiment | 클러스터 레벨 sentiment | 1h |
| 3 | 다중 게임 | 게임별 별도 CSV | 3~4h |
| 4 | 인증 | link-only 공유 | 2h |
| 5 | 정량 평가 지표 | spot check + silhouette | 4~6h |
| 6 | 모바일 UI | 데스크톱 전용 | 2~3h |
| 7 | 콜드 스타트 | precomputed artifacts + warm up | 유료 플랜 or 이관 |
| 8 | Slack 알림 | Jira bulk import | 1.5~2h |

모든 항목은 **의식적 trade-off** 이며, 3 시간 제한 해제 시 우선순위는 **1 → 8 → 5 → 2** 순(실용성 기준) 입니다.
