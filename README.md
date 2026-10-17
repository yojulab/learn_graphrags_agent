# GraphRAG Agent 🕸️
> **귀멸의 칼날 시즌 1 지식그래프 기반 Dual-Backend 하이브리드 검색 에이전트**

본 프로젝트는 위키피디아 시즌 1 줄거리를 기반으로 지식그래프를 구축하고, **벡터 검색(Vector Search)**과 **그래프 순회(Graph Traversal)**를 결합한 하이브리드 검색을 통해 정밀한 자연어 질의응답을 제공하는 GraphRAG 에이전트 시스템입니다.

---

## 📌 주요 사항 요약

- **듀얼 그래프 DB 백엔드 지원**:
  - **Neo4j 5+**: Bolt 프로토콜, HNSW 코사인 유사도 벡터 인덱스 (`bge-m3:567m`), 1~2 Hop 동적 그래프 확장
  - **PostgreSQL 16+ & Apache AGE 1.5+**: `ag_catalog` 기반 openCypher in SQL, In-Database 관계형 벡터 테이블, SQL JOIN 멀티모델 융합
- **10대 골든 룰 및 가드레일**: 읽기 전용 쿼리 안전성(`MATCH` 전용), 19개 정규화 인물/도깨비 노드, 24개 표준 관계, `<think>` 태그 정제, 데이터 적재 멱등성 보장
- **표준 MCP 서버 탑재**: `graphrag-db-inspector` MCP 도구를 통해 Neo4j, Apache AGE, MongoDB 상태 실시간 점검 및 안전 쿼리 지원
- **인터랙티브 웹 대시보드**: FastAPI + Tailwind-free 모던 웹 UI를 통한 실시간 버전 전환(v1~v4) 및 원클릭 데이터 수집

---

## 🗺️ 버전별 상세 가이드 및 아키텍처 링크

각 버전의 **상세 실행 다이어그램**, **실행 파일 순서(설명/참조/입출력 표)**, **적재 데이터 확인 예제**는 각 폴더의 상세 `README.md`에 분산 정리되어 있습니다:

| 버전 | 폴더 링크 | 백엔드 DB | 검색 아키텍처 | 상세 가이드 링크 |
| :---: | :--- | :--- | :--- | :---: |
| **v1** | [`src/baseline_v1/`](file:///apps/learn_graphrags_agent/src/baseline_v1/) | Neo4j 5 | 기본 Text2Cypher 베이스라인 (0~1 hop 단일 매칭) | 📖 [v1 README.md](file:///apps/learn_graphrags_agent/src/baseline_v1/README.md) |
| **v2** | [`src/metadata_filter_v2/`](file:///apps/learn_graphrags_agent/src/metadata_filter_v2/) | Neo4j 5 | 메타데이터(호흡/에피소드) 필터링 + 벡터 RAG + LLM Reranking | 📖 [v2 README.md](file:///apps/learn_graphrags_agent/src/metadata_filter_v2/README.md) |
| **v3** | [`src/neo4j_hybrid_v3/`](file:///apps/learn_graphrags_agent/src/neo4j_hybrid_v3/) | Neo4j 5 | 1024차원 HNSW 벡터 인덱스 + 1~2 Hop 동적 그래프 확장 | 📖 [v3 README.md](file:///apps/learn_graphrags_agent/src/neo4j_hybrid_v3/README.md) |
| **v4** | [`src/apache_age_v4/`](file:///apps/learn_graphrags_agent/src/apache_age_v4/) | PostgreSQL 16 & Apache AGE | In-DB SQL 벡터 검색 + openCypher in SQL 멀티모델 | 📖 [v4 README.md](file:///apps/learn_graphrags_agent/src/apache_age_v4/README.md) |

---

## 🛠️ 버전별 핵심 기능 비교

| 항목 | baseline (v1) | metadata_filter (v2) | neo4j_hybrid (v3) | apache_age (v4) |
| :--- | :---: | :---: | :---: | :---: |
| **백엔드 DB** | Neo4j 5 | Neo4j 5 | Neo4j 5 | PostgreSQL 16+ & Apache AGE |
| **그래프 모델** | Native Property Graph | Native Property Graph | Native Property Graph | Relational Property Graph (`ag_catalog`) |
| **쿼리 언어** | Neo4j Cypher | Neo4j Cypher | Neo4j Cypher | openCypher in SQL (`cypher(...)`) |
| **벡터 인덱스** | 미지원 (순수 Cypher) | 벡터 + 메타데이터 필터링 | 26개 HNSW 코사인 인덱스 | In-Database 관계형 벡터 테이블 |
| **그래프 순회** | 0-hop (단일 매칭) | 1-hop 정적 필터링 | 1~2 hop 동적 하이브리드 확장 | 1~2 hop AGE 하이브리드 순회 |
| **RDBMS 결합** | 불가 (그래프 단독) | 불가 (그래프 단독) | 불가 (그래프 단독) | **완전 지원** (SQL JOIN + Cypher 동시 질의) |

---

## ⚙️ 사전 요구사항 및 인프라 실행

### 1. 사전 요구사항
- **Python 3.12+**
- **Docker & Docker Compose**
- **Ollama 또는 OpenAI API**

```bash
# 의존성 설치
uv sync  # 또는 pip install -r requirements.txt

# Playwright 브라우저 설치 (데이터 수집 및 UI 테스트용)
playwright install chromium
```

### 2. 환경 변수 (`.env`)
프로젝트 루트의 `.env`에 DB 및 LLM 접속 정보를 설정합니다:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=admin123
NEO4J_DATABASE=db-graphrag-agent

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=admin123
POSTGRES_DB=db-graphrag-agent

LLM_MODEL=sam860/exaone-4.0:1.2b-thinking-Q8_0
EMBEDDING_MODEL=bge-m3:567m
MODEL_API_URL=http://localhost:11434/v1
```

### 3. Docker 인프라 기동
```bash
cd dockers
docker-compose up -d
cd ..
```
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) (Bolt: `localhost:7687`)
- **PostgreSQL / Apache AGE**: `localhost:5432`
- **Ollama API**: [http://localhost:11434](http://localhost:11434)

---

## ⚡ 빠른 시작 (Quick Start)

### Step 1: 데이터 수집 & 정규화
```bash
# Playwright 기반 정밀 데이터 수집
python3 src/utils/playwright_scraper.py
```

### Step 2: 지식그래프 적재 (원하는 백엔드 선택)
```bash
# [Neo4j] v3 하이브리드 그래프 & HNSW 인덱스 적재
python3 src/neo4j_hybrid_v3/2_ingest_data_v3.py

# [Apache AGE] v4 멀티모델 그래프 & 벡터 테이블 적재
python3 src/apache_age_v4/2_ingest_data_v4.py
```

### Step 3: GraphRAG 에이전트 CLI 질의
```bash
# Neo4j Production Hybrid 실행
python3 src/neo4j_hybrid_v3/3_graphrag_agent_v3.py

# Apache AGE Multi-Model 실행
python3 src/apache_age_v4/3_graphrag_agent_v4.py
```

---

## 🖥️ 웹 서비스 & 대시보드 (`src/web_app/`)

실시간 버전 전환(v1~v4), 추천 질문 칩, 그래프 통계 대시보드를 제공하는 FastAPI 웹 애플리케이션입니다:

```bash
uvicorn src.web_app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **대시보드 URL**: [http://localhost:8000](http://localhost:8000)
- **API 엔드포인트**:
  - `GET /api/stats`: 지식그래프 총 노드/관계 통계
  - `POST /api/query`: 버전별(v1/v2/v3/v4) 하이브리드 검색 질의
  - `POST /api/scrape`: 웹 대시보드 내 원클릭 데이터 수집

---

## 🔧 유틸리티 및 MCP DB Inspector (`src/utils/`)

- [`src/utils/mcp_db_server.py`](file:///apps/learn_graphrags_agent/src/utils/mcp_db_server.py): Neo4j, Apache AGE, MongoDB 3대 DB 상태를 실시간 진단하고 읽기 전용 쿼리를 수행하는 Model Context Protocol 서버
  ```bash
  python3 src/utils/mcp_db_server.py --test
  ```
- [`src/utils/check_indexes.py`](file:///apps/learn_graphrags_agent/src/utils/check_indexes.py): Neo4j HNSW 벡터 인덱스 구축 상태 점검
- [`src/utils/verify_full_pipeline.py`](file:///apps/learn_graphrags_agent/src/utils/verify_full_pipeline.py): Flow 1~4 전체 파이프라인 일괄 E2E 자동 검증기
  ```bash
  python3 src/utils/verify_full_pipeline.py
  ```

---

## 📁 프로젝트 폴더 구조

```text
/apps/learn_graphrags_agent
├── .env                              # 전체 중앙 환경변수 (SSOT)
├── config.py                         # 중앙 설정 로더
├── mcp_config.json                   # Multi-Model DB Inspector MCP 서버 설정
├── docs/                             # 사양서 및 아키텍처 문서
│   ├── PRD.md                        # 제품 요구사항 정의서
│   └── agent/
│       ├── harness_prompt.md         # 하네스 프롬프트 규격서
│       └── system_prompt.md          # 시스템 프롬프트 규격서
├── output/                           # 정제된 JSON 지식그래프 저장소
├── dockers/                          # Docker Compose 인프라 환경
└── src/                              # 업무/버전별 소스 코드
    ├── baseline_v1/                  # [v1] Text2Cypher 베이스라인 (README.md 포함)
    ├── metadata_filter_v2/           # [v2] 메타데이터 필터링 & Reranking (README.md 포함)
    ├── neo4j_hybrid_v3/              # [v3] Neo4j HNSW 벡터+그래프 하이브리드 (README.md 포함)
    ├── apache_age_v4/                # [v4] PostgreSQL 16+ & Apache AGE 멀티모델 (README.md 포함)
    ├── web_app/                      # FastAPI 웹 대시보드
    └── utils/                        # 공통 수집기, 검증기, MCP 서버
```

---

## 📚 관련 핵심 문서 링크

- [PRD.md](file:///apps/learn_graphrags_agent/docs/PRD.md): 상세 제품 사양서 및 아키텍처 분석
- [harness_prompt.md](file:///apps/learn_graphrags_agent/docs/agent/harness_prompt.md): 4단계 에이전트 하네스 프롬프트 표준 규격
- [system_prompt.md](file:///apps/learn_graphrags_agent/docs/agent/system_prompt.md): 런타임 시스템 프롬프트 및 가드레일 지침
- [AGENTS.md](file:///apps/learn_graphrags_agent/AGENTS.md): 에이전트 행동 강령 및 표준 지침
- [GEMINI.md](file:///apps/learn_graphrags_agent/GEMINI.md): 하네스 헌장 및 10대 골든 룰
