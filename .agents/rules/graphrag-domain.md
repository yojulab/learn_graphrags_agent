# GraphRAG Agent 도메인 및 파이프라인 엔지니어링 가이드
## (Demon Slayer Season 1 Knowledge Graph & Dual-Backend GraphRAG)

- **문서 참조**: `docs/PRD.md` (Demon Slayer Season 1 Knowledge Graph PRD)
- **프로젝트 명칭**: **`learn_graphrags_agent`**
- **도메인 대상**: 귀멸의 칼날 시즌 1 (1화~26화) 위키피디아 에피소드 줄거리 데이터 기반 지식그래프 구축 및 자연어 질의응답
- **핵심 목표**: 비정형 줄거리 텍스트로부터 정형 지식그래프(KG)를 추출하여 듀얼 그래프 데이터베이스(**Neo4j** & **Apache AGE on PostgreSQL**)에 적재하고, 벡터 유사도 검색 + 그래프 확장 + 자연어 Cypher 생성을 결합한 고정밀 하이브리드 검색(Hybrid Retrieval) 에이전트를 제공함.
- **핵심 런타임 및 인프라**:
  - Python 3.12+ / `uv`
  - Neo4j 5+ (Bolt protocol, 포트 7687, HTTP 7474)
  - PostgreSQL 16+ & Apache AGE 1.5+ (포트 5432, `ag_catalog`)
  - Ollama LLM (`sam860/exaone-4.0:1.2b-thinking-Q8_0`) & Embedding (`bge-m3:567m`, 1024차원)
  - FastAPI 웹 탐색기 및 대시보드 (`src/app/main.py`)

---

## 1. 듀얼 백엔드 아키텍처 및 4단계 버전(v1~v4) 파이프라인

### 1-1. 버전별 파이프라인 진화 매트릭스

| 버전 | 명칭 및 백엔드 DB | 그래프 모델 | 질의 언어 | 벡터 검색 | 그래프 확장 | RDBMS 결합 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Baseline (Neo4j) | Native Property Graph | Neo4j Cypher | 미지원 (순수 Cypher) | 0 hop (단일 매칭) | 불가 |
| **v2** | Metadata Filter (Neo4j) | Native Property Graph | Neo4j Cypher | 벡터 검색 + 메타 필터링 | 1 hop 정적 확장 | 불가 |
| **v3** | Production Hybrid (Neo4j) | Native Property Graph | Neo4j Cypher | HNSW 노드/간선 벡터 인덱스 | 1~2 hop 동적 확장 | 불가 |
| **v4** | Multi-Model (Apache AGE) | Relational Property Graph | openCypher in SQL | PostgreSQL 벡터 테이블 / Cosine | 1~2 hop AGE 순회 | **가능 (SQL JOIN+Cypher)** |

### 1-2. 공통 3단계 실행 흐름 (3-Step Pipeline Flow)

각 버전(v1~v4)은 일관된 3단계 스크립트 실행 구조를 따릅니다:

1. **Step 1: Data Preparation (`src/v{N}/1_prepare_data_v{N}.py`)**
   - 위키피디아 1~26화 줄거리 수집 및 BeautifulSoup4 / Playwright 정제.
   - Pydantic v2 기반 스키마 검증 (`Node`, `Relationship`, `GraphResponse`).
   - 한국어 표준 표기 정규화 (예: `카마도 탄지로`, `토미오카 기유`, `키부츠지 무잔`).
   - 표준 노드 라벨(`인간`, `도깨비`) 및 24개 관계 타입 검증.
   - 산출물: `output/raw_data_v{N}.json`, `output/knowledge_graph_v{N}.json`, `output/statistics_v{N}.json`.

2. **Step 2: Data Ingestion (`src/v{N}/2_ingest_data_v{N}.py`)**
   - **v1~v3 (Neo4j)**: 기존 DB 삭제/초기화로 멱등성 보장, `bge-m3` 1024차원 노드/관계 임베딩 생성 및 적재, HNSW 벡터 인덱스 생성 (`entity_embeddings_*`, `rel_embeddings_*`).
   - **v4 (Apache AGE)**: `LOAD 'age';` 및 `SET search_path = ag_catalog, "$user", public;`, `drop_graph('demon_slayer', true)` / `create_graph('demon_slayer')`, 정점 라벨(`인간`, `도깨비`) 및 24개 간선 라벨 등록, `SELECT * FROM cypher(...)`로 정점/간선 생성, PostgreSQL 임베딩 테이블 적재 및 코사인 유사도 검색 인덱스 구성.

3. **Step 3: GraphRAG Agent (`src/v{N}/3_graphrag_agent_v{N}.py`)**
   - **Query Classifier**: 질문 유형 판별 (`SINGLE_ENTITY`, `RELATIONSHIP`, `EPISODE_SPECIFIC`, `GENERAL`).
   - **Vector Seed Retrieval**: 코사인 유사도 상위 Top-K 시드 정점/관계 추출.
   - **Graph Expansion**: 시드 정점 중심 1~2 hop 서브그래프 동적 순회.
   - **Clean LLM Parser**: `<think>...</think>` 추론 태그 제거 및 쿼리/답변 추출.
   - **Text2Cypher / openCypher Executor**: 읽기 전용 쿼리 실행 및 최종 한국어 개조식 답변 합성.

---

## 2. Apache AGE (PostgreSQL Multi-Model Graph) 상세 규격

### 2-1. Apache AGE 핵심 원리
- PostgreSQL 관계형 데이터베이스 상에서 그래프 모델을 지원하는 확장 프로그램 (`ag_catalog`).
- openCypher 질의를 PostgreSQL SQL 함수형 래퍼(`cypher('graph_name', $$ ... $$) as (...)`) 형태로 호출.
- 결과값은 상위 JSON 호환 포맷인 `agtype`으로 반환되며, `psycopg2` 또는 `asyncpg`를 통해 Python 데이터형으로 파싱.

### 2-2. 필수 AGE 초기화 DDL
```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- 멱등 그래프 생성
SELECT drop_graph('demon_slayer', true);
SELECT create_graph('demon_slayer');

-- 정점(Vertex) 라벨 등록
SELECT create_vlabel('demon_slayer', '인간');
SELECT create_vlabel('demon_slayer', '도깨비');

-- 24개 간선(Edge) 라벨 등록 (FIGHTS, PROTECTS, TRAINS, ...)
SELECT create_elabel('demon_slayer', 'FIGHTS');
SELECT create_elabel('demon_slayer', 'PROTECTS');
-- (기타 22개 간선 라벨 등록)
```

### 2-3. openCypher 질의 패턴
```sql
SELECT * FROM cypher('demon_slayer', $$
    MATCH (n:인간 {name: '카마도 탄지로'})-[r]-(m)
    RETURN n.name, type(r), m.name
$$) as (source agtype, rel agtype, target agtype);
```

---

## 3. 환경 변수 및 설정 규격 (`.env` & `config.py`)

모든 파이프라인과 에이전트는 루트 `.env`와 `config.py`의 중앙 설정을 엄격히 참조합니다:

```env
# Infrastructure & Profiles
PROJECT_NAME=graphrag_agent
COMPOSE_PROFILES=neo4j,postgres,mongo,ollama,app

# Neo4j Database (v1 ~ v3)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=admin123
NEO4J_DATABASE=db-graphrag-agent

# PostgreSQL + Apache AGE (v4)
AGE_IMAGE_TAG=latest
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=db_graphrag_agent
AGE_GRAPH_NAME=demon_slayer

# MongoDB
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USER=admin
MONGO_PASSWORD=admin123
MONGO_DB=db_graphrag_agent

# LLM & Embedding Server
LLM_MODEL=sam860/exaone-4.0:1.2b-thinking-Q8_0
EMBEDDING_MODEL=bge-m3:567m
MODEL_API_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
AI_OLAMA_CPU=8

# Hybrid Retrieval Settings
USE_HYBRID_RETRIEVAL=true
VECTOR_TOP_K=5
GRAPH_EXPANSION_DEPTH=2

# Vector Indexes & Schema Grounding
VECTOR_INDEX_NODE=entity_embeddings
VECTOR_INDEX_RELATIONSHIP_PREFIX=rel_embeddings
EMBEDDING_DIMENSION=1024
NODE_LABELS=인간,도깨비
RELATIONSHIP_TYPES=FIGHTS,PROTECTS,TRAINS,TRAINS_WITH,KNOWS,FAMILY_OF,SIBLING_OF,ALLY_OF,ENEMY_OF,DEFEATS,SAVES,RESCUES,MEETS,ENCOUNTERS,GUIDES,ATTACKS,DEFENDS,TRANSFORMS,JOINS,SUPPORTS,REUNITES_WITH,BATTLES,HEALS,TEACHES
```

---

## 4. 실행 및 검증 가이드라인

### 4-1. 인프라 컨테이너 실행
```bash
cd dockers
docker-compose --profile neo4j --profile postgres --profile ollama up -d
cd ..
```

### 4-2. Neo4j v3 파이프라인 실행
```bash
uv run src/v3/1_prepare_data_v3.py
uv run src/v3/2_ingest_data_v3.py
uv run src/v3/3_graphrag_agent_v3.py
uv run src/utils/check_indexes.py
```

### 4-3. Apache AGE v4 파이프라인 실행
```bash
uv run src/v4/1_prepare_data_v4.py
uv run src/v4/2_ingest_data_v4.py
uv run src/v4/3_graphrag_agent_v4.py
```

### 4-4. PostgreSQL AGE psql 직접 검증
```bash
docker exec -it graphrag_agent_db_postgres psql -U admin -d db_graphrag_agent -c "
LOAD 'age';
SET search_path = ag_catalog, '\$user', public;
SELECT * FROM cypher('demon_slayer', \$\$
    MATCH (a:인간 {name: '카마도 탄지로'})-[r]->(b)
    RETURN a.name, type(r), b.name
\$\$) as (source agtype, rel agtype, target agtype);
"
```
