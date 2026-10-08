# GraphRAG Agent 🕸️

애니메이션 줄거리를 AI로 분석하여 지식 그래프를 만들고, 하이브리드 검색(벡터 + 그래프)으로 질문에 답하는 GraphRAG 에이전트입니다.

---

## 🎯 프로젝트 개요

**하이브리드 검색 파이프라인**: 벡터 검색 → 그래프 확장 → Cypher 쿼리 → 결과 병합

1. **데이터 준비** - 위키피디아 스크래핑 → AI 개체/관계 추출 → JSON 저장
2. **그래프 생성** - JSON → Neo4j 지식그래프 + 벡터 임베딩 저장
3. **하이브리드 검색** - 자연어 질문 → 벡터 유사도 검색 → 그래프 순회 → Cypher 쿼리 생성 → 답변

📄 **상세 사양서**: 자세한 프로젝트 분석 및 아키텍처는 [PRD.md](file:///apps/learn_graphrags_agent/PRD.md)를 참조하세요.

---

## 📋 사전 요구사항

- **Python 3.12+**
- **Neo4j Database** (Docker Compose 권장)
- **Ollama 또는 OpenAI API**

---

## 🚀 실행 가이드 (Execution Guide)

본 프로젝트는 **데이터 수집(Playwright Headless) → Neo4j 그래프 적재 및 벡터 인덱싱 → GraphRAG 하이브리드 검색 CLI → FastAPI 웹 대시보드 → E2E 파이프라인 검증**의 전 과정을 모듈화하여 지원합니다.

---

### 1. 환경 설정 및 의존성 설치

#### 1.1 Python 패키지 설치
`uv` 또는 `pip`를 사용하여 필수 의존성을 설치합니다:

```bash
# uv 사용 시 (권장)
uv sync

# 또는 pip 사용 시
pip install -r requirements.txt
```

#### 1.2 Playwright Headless 브라우저 설치
웹 스크래핑 및 E2E UI 검증을 위해 Chromium 브라우저 바이너리를 설치합니다:

```bash
# Playwright Chromium 브라우저 설치
playwright install chromium

# (Linux/Docker 환경) 필수 시스템 라이브러리 설치
playwright install-deps chromium
```

---

### 2. 환경 변수 설정 (`.env`)

프로젝트 루트의 `.env` 파일에서 Neo4j 데이터베이스 및 AI 모델 연결 정보를 설정합니다:

```env
# Neo4j 데이터베이스 연결
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=admin123
NEO4J_DATABASE=db-graphrag-agent

# LLM 및 임베딩 모델 (Ollama 또는 OpenAI)
LLM_MODEL=sam860/exaone-4.0:1.2b-thinking-Q8_0
EMBEDDING_MODEL=bge-m3:567m
MODEL_API_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama

# 하이브리드 검색 설정
USE_HYBRID_RETRIEVAL=true
VECTOR_TOP_K=5
GRAPH_EXPANSION_DEPTH=2
```

---

### 3. 인프라 실행 (Docker Compose)

`dockers` 폴더에서 Docker Compose로 Neo4j 및 Ollama 인프라를 실행합니다:

```bash
cd dockers

# 전체 서비스 실행 (Neo4j, Postgres AGE, Mongo, Ollama, App)
docker-compose up -d

# 또는 특정 서비스만 선별 실행 (예: Neo4j 전용)
docker-compose --profile neo4j up -d

cd ..
```

> **서비스 접속 정보**:
> - **Neo4j Browser**: `http://localhost:7474` (Bolt: `localhost:7687`)
> - **Ollama API**: `http://localhost:11434`

---

### 4. 파이프라인 단계별 실행 (Step-by-Step)

#### Step 1: 데이터 수집 및 개체 추출 (`1_prepare_data`)
위키피디아 시즌 1 줄거리 데이터를 수집하고 LLM을 통해 지식그래프 노드/관계를 정형화합니다:

```bash
# [방법 A] Playwright Headless 기반 정밀 수집기 단독 실행
python3 src/utils/playwright_scraper.py

# [방법 B] v3 파이프라인 (수집 + 정규화된 한국어 개체/관계 추출)
python3 src/v3/1_prepare_data_v3.py
# (uv 사용 시: uv run src/v3/1_prepare_data_v3.py)
```
- **출력 아티팩트**: `output/raw_data.json`, `output/knowledge_graph_v3.json`

#### Step 2: Neo4j 지식그래프 적재 및 벡터 인덱싱 (`2_ingest_data`)
추출된 JSON 데이터를 Neo4j 데이터베이스에 적재하고 HNSW 벡터 인덱스를 구축합니다:

```bash
python3 src/v3/2_ingest_data_v3.py
# (uv 사용 시: uv run src/v3/2_ingest_data_v3.py)
```
- **주요 동작**: 이전 데이터 초기화 → 노드 및 관계 생성 → 1024차원 임베딩 저장 → 벡터 인덱스(`entity_embeddings_*`, `rel_embeddings_*`) 자동 생성

#### Step 3: GraphRAG 에이전트 CLI 질의응답 (`3_graphrag_agent`)
터미널에서 자연어 질문을 입력하여 하이브리드 검색 답변을 생성합니다:

```bash
# v3.0 최신 하이브리드 검색 (벡터 검색 + 그래프 1~2hop + Text2Cypher)
python3 src/v3/3_graphrag_agent_v3.py

# v2.0 메타데이터 필터링 + LLM Reranking 버전
python3 src/v2/3_graphrag_agent_v2.py

# v1.0 기본 Text2Cypher 베이스라인 버전
python3 src/v1/3_graphrag_agent_v1.py
```

---

### 5. 웹 서비스 & 대시보드 실행

FastAPI 기반의 인터랙티브 웹 대시보드를 구동하여 브라우저에서 편리하게 질문하고 결과를 시각적으로 확인합니다:

```bash
# FastAPI 개발 서버 실행 (포트 8000)
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 main.py 직접 실행
python3 src/app/main.py
```

- **웹 대시보드 URL**: `http://localhost:8000`
- **주요 기능**:
  - 💬 **대화형 질의응답 창**: 추천 질문 칩 클릭 또는 자연어 질문 입력
  - 🎛️ **버전 전환 셀렉터**: v3(하이브리드), v2(메타데이터), v1(Text2Cypher) 실시간 전환
  - 📊 **실시간 통계 카드**: 총 노드/관계 수, 에피소드 수 등 지식그래프 상태 표시
  - 🕸️ **원클릭 데이터 수집**: 대시보드 내에서 Playwright Headless 수집 트리거 지원

---

### 6. Playwright Headless 전체 파이프라인 E2E 검증

데이터 수집기, Web UI 대시보드 인터랙션, PRD 스키마 무결성을 자동으로 일괄 검증합니다:

```bash
python3 src/utils/test_pipeline_playwright.py
```

- **검증 항목**:
  1. `[Test 1] Playwright Headless Scraper Test`: 위키피디아 에피소드 파싱 및 공백 보존 검증
  2. `[Test 2] Playwright Web UI E2E Test`: 대시보드 DOM 렌더링, 타이틀, 폼 입력 및 API 인터랙션 검증
  3. `[Test 3] Schema Integrity Test`: PRD 정의 `NODE_LABELS`(`인간`, `도깨비`) 및 24종 `RELATIONSHIP_TYPES` 100% 일치 검증
- **결과 보고서**: `output/playwright_test_report.json`으로 자동 저장

---

### 7. 유틸리티 및 인덱스 점검

Neo4j에 정상 구축된 HNSW 벡터 인덱스 목록을 확인합니다:

```bash
python3 src/utils/check_indexes.py
```

---

## 📁 디렉토리 구조 (버전별 폴더 구성)

```text
/apps/learn_graphrags_agent
├── .env                              # 전체 중앙 환경변수 관리
├── config.py                         # .env 기반 설정을 로딩하는 모듈
├── PRD.md                            # 요구사항 및 상세 분석 문서
├── README.md                         # 실행 및 프로젝트 설명서
├── dockers/                          # Docker 및 컨테이너 관련 파일
│   ├── .env
│   ├── docker-compose.yml
│   ├── Dockerfile.ollama
│   └── Dockerfile.fullstack
├── output/                           # 파이프라인 추출 JSON 데이터 저장소
├── docs/
│   └── agent/
│       ├── harness_prompt.md         # 에이전트 하네스 프롬프트 규격서
│       └── system_prompt.md          # 시스템 프롬프트 규격서
└── src/                              # 버전별 실행 코드 폴더
    ├── app/                          # GraphRAG 웹 애플리케이션 및 대시보드
    │   ├── __init__.py
    │   └── main.py
    ├── v1/                           # v1.0 baseline 스크립트
    │   ├── 1_prepare_data_v1.py
    │   ├── 2_ingest_data_v1.py
    │   └── 3_graphrag_agent_v1.py
    ├── v2/                           # v2.0 메타데이터 필터링 벡터 RAG 스크립트
    │   ├── 1_prepare_data_v2.py
    │   ├── 2_ingest_data_v2.py
    │   └── 3_graphrag_agent_v2.py
    ├── v3/                           # v3.0 최신 하이브리드 검색 GraphRAG 스크립트
    │   ├── 1_prepare_data_v3.py
    │   ├── 2_ingest_data_v3.py
    │   └── 3_graphrag_agent_v3.py
    └── utils/                        # 유틸리티 스크립트
        ├── check_indexes.py          # Neo4j 벡터 인덱스 점검
        ├── playwright_scraper.py     # Playwright Headless 데이터 수집기
        └── test_pipeline_playwright.py # Playwright Headless E2E 통합 검증 스크립트
```

---

## 🛠️ 버전별 주요 차이점

| 버전 | 데이터 추출 (`1_prepare_data`) | 데이터 인제스천 (`2_ingest_data`) | GraphRAG 에이전트 (`3_graphrag_agent`) |
|---|---|---|---|
| **v1** | 기본 OpenAI 스킴 추출 | 기본 Neo4j 그래프 생성 | Text2Cypher 기본 쿼리 생성 |
| **v2** | 정형 데이터 검증 추가 | 노드/관계 벡터 임베딩 생성 | 메타데이터 기반 벡터 검색 + LLM Reranking |
| **v3 (권장)** | 정규화된 한국어 이름 + 릴레이션 마스터 매핑 | HNSW 벡터 인덱스 자동 생성 | **하이브리드 검색** (벡터 + 그래프 1~2hop + Text2Cypher + Think 태그 정화) |

---

## 🎮 예시 질문

```text
"카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?"
"토미오카 기유와 관계있는 모든 캐릭터를 알려줘."
"루이와 싸운 캐릭터는 누구야?"
"3번 이상 등장한 관계 타입은?"
```

---

## 📚 관련 문서 및 사양서

- [PRD.md](file:///apps/learn_graphrags_agent/PRD.md): 상세 제품 사양 및 아키텍처 분석
- [docs/agent/harness_prompt.md](file:///apps/learn_graphrags_agent/docs/agent/harness_prompt.md): 에이전트 하네스 프롬프트 표준 규격서
- [.agent/rules/rule-lock-fullstack.md](file:///apps/learn_graphrags_agent/.agent/rules/rule-lock-fullstack.md): 프로젝트 규칙 및 아키텍처 락
