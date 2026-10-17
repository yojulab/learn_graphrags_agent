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
위키피디아 시즌 1 줄거리 데이터를 수집하고 LLM/정규화기를 통해 지식그래프 노드/관계를 정형화합니다:

```bash
# [방법 A] Playwright Headless 기반 정밀 수집기 단독 실행
python3 src/utils/playwright_scraper.py

# [방법 B] Neo4j 하이브리드 파이프라인 데이터 준비
python3 src/neo4j_hybrid/1_prepare_data_v3.py

# [방법 C] Apache AGE 멀티모델 파이프라인 데이터 준비
python3 src/apache_age/1_prepare_data_v4.py
```
- **출력 아티팩트**: `output/raw_data.json`, `output/knowledge_graph_v3.json`, `output/knowledge_graph_v4.json`

#### Step 2: 지식그래프 적재 및 벡터 인덱싱 (`2_ingest_data`)
추출된 JSON 데이터를 그래프 데이터베이스에 적재하고 벡터 인덱스를 구축합니다:

```bash
# [백엔드 A] Neo4j 5+ 그래프 적재 및 HNSW 벡터 인덱스 생성
python3 src/neo4j_hybrid/2_ingest_data_v3.py

# [백엔드 B] PostgreSQL 16+ & Apache AGE Multi-Model 적재 및 벡터 테이블 생성
python3 src/apache_age/2_ingest_data_v4.py
```
- **주요 동작**: 이전 데이터 멱등 초기화 → 노드 및 관계 생성 → 1024차원 `bge-m3` 임베딩 저장 → 벡터 인덱스/테이블 구축

#### Step 3: GraphRAG 에이전트 CLI 질의응답 (`3_graphrag_agent`)
터미널에서 자연어 질문을 입력하여 하이브리드 검색 답변을 생성합니다:

```bash
# Apache AGE Multi-Model (openCypher in SQL + In-Database 벡터 검색)
python3 src/apache_age/3_graphrag_agent_v4.py

# Neo4j 최신 하이브리드 검색 (벡터 검색 + 그래프 1~2hop + Cypher)
python3 src/neo4j_hybrid/3_graphrag_agent_v3.py

# 메타데이터 필터링 + LLM Reranking 버전
python3 src/metadata_filter/3_graphrag_agent_v2.py

# 기본 Text2Cypher 베이스라인 버전
python3 src/baseline/3_graphrag_agent_v1.py
```

---

### 5. 웹 서비스 & 대시보드 실행

FastAPI 기반의 인터랙티브 웹 대시보드를 구동하여 브라우저에서 편리하게 질문하고 결과를 시각적으로 확인합니다:

```bash
# FastAPI 개발 서버 실행 (포트 8000)
uvicorn src.web_app.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 main.py 직접 실행
python3 src/web_app/main.py
```

- **웹 대시보드 URL**: `http://localhost:8000`
- **주요 기능**:
  - 💬 **대화형 질의응답 창**: 추천 질문 칩 클릭 또는 자연어 질문 입력
  - 🎛️ **버전 전환 셀렉터**: v4(Apache AGE 멀티모델), v3(Neo4j 하이브리드), v2(메타데이터), v1(Text2Cypher) 실시간 전환
  - 📊 **실시간 통계 카드**: 총 노드/관계 수, 에피소드 수 등 지식그래프 상태 동적 표시
  - 🕸️ **원클릭 데이터 수집**: 대시보드 내에서 Playwright Headless 수집 트리거 지원

---

### 6. 전체 파이프라인 E2E 검증 (Playwright Headless)

데이터 수집기, 듀얼 DB 정합성, Dual GraphRAG 에이전트, Web UI 대시보드 인터랙션을 일괄 검증합니다:

```bash
# 전체 파이프라인 흐름(Flow 1~4) 일괄 검증
python3 src/utils/verify_full_pipeline.py

# Playwright 단위/통합 테스트
python3 src/utils/test_pipeline_playwright.py
```

---

### 7. 유틸리티 및 인덱스 점검

Neo4j에 정상 구축된 HNSW 벡터 인덱스 목록을 확인합니다:

```bash
python3 src/utils/check_indexes.py
```

---

## 📁 디렉토리 구조 (업무별 폴더 구성)

```text
/apps/learn_graphrags_agent
├── .env                              # 전체 중앙 환경변수 관리
├── config.py                         # .env 기반 설정을 로딩하는 모듈
├── PRD.md                            # 요구사항 및 상세 분석 문서 (docs/PRD.md)
├── README.md                         # 실행 및 프로젝트 설명서
├── dockers/                          # Docker 및 컨테이너 관련 파일
│   ├── .env
│   ├── docker-compose.yml
│   ├── Dockerfile.ollama
│   ├── Dockerfile.fullstack
│   └── postgres/init/01_init_age.sql
├── output/                           # 파이프라인 추출 JSON 데이터 저장소
├── docs/
│   ├── PRD.md                        # 제품 요구사항 정의서
│   └── agent/
│       ├── harness_prompt.md         # 에이전트 하네스 프롬프트 규격서
│       └── system_prompt.md          # 시스템 프롬프트 규격서
└── src/                              # 업무별 실행 코드 폴더
    ├── baseline/                     # [구 v1] Text2Cypher 베이스라인 파이프라인
    │   ├── 1_prepare_data_v1.py
    │   ├── 2_ingest_data_v1.py
    │   └── 3_graphrag_agent_v1.py
    ├── metadata_filter/              # [구 v2] 메타데이터 필터링 및 리랭킹 파이프라인
    │   ├── 1_prepare_data_v2.py
    │   ├── 2_ingest_data_v2.py
    │   └── 3_graphrag_agent_v2.py
    ├── neo4j_hybrid/                 # [구 v3] Neo4j 벡터+그래프 1~2hop 하이브리드 RAG
    │   ├── 1_prepare_data_v3.py
    │   ├── 2_ingest_data_v3.py
    │   └── 3_graphrag_agent_v3.py
    ├── apache_age/                   # [신규 v4] PostgreSQL 16+ & Apache AGE Multi-Model RAG
    │   ├── 1_prepare_data_v4.py
    │   ├── 2_ingest_data_v4.py
    │   └── 3_graphrag_agent_v4.py
    ├── web_app/                      # FastAPI 웹 대시보드 및 REST API
    │   ├── __init__.py
    │   └── main.py
    └── utils/                        # 공통 유틸리티 및 검증 도구
        ├── check_indexes.py          # Neo4j 벡터 인덱스 점검
        ├── playwright_scraper.py     # Playwright Headless 데이터 수집기
        ├── test_pipeline_playwright.py # Playwright 단위/통합 테스트
        └── verify_full_pipeline.py   # 전체 흐름(Flow 1~4) E2E 검증기
```

---

## 🛠️ 버전별 주요 차이점

| 항목 | baseline (v1) | metadata_filter (v2) | neo4j_hybrid (v3) | apache_age (v4) |
|---|---|---|---|---|
| **백엔드 DB** | Neo4j 5 | Neo4j 5 | Neo4j 5 | PostgreSQL 16+ & Apache AGE |
| **그래프 모델** | Native Property Graph | Native Property Graph | Native Property Graph | Relational Property Graph (`ag_catalog`) |
| **쿼리 언어** | Neo4j Cypher | Neo4j Cypher | Neo4j Cypher | openCypher in SQL (`cypher(...)`) |
| **벡터 검색** | 미지원 (순수 Cypher) | 벡터 + 메타데이터 필터링 | HNSW 노드/관계 벡터 인덱스 | In-Database Cosine 테이블 검색 |
| **그래프 순회** | 0-hop (단일 매칭) | 1-hop 정적 순회 | 1~2 hop 동적 하이브리드 확장 | 1~2 hop AGE 하이브리드 순회 |
| **RDBMS 결합** | 불가 (그래프 전용) | 불가 (그래프 전용) | 불가 (그래프 전용) | **가능** (SQL JOIN + Cypher 동시 질의) |

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
