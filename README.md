# GraphRAG Agent 🕸️

애니메이션 줄거리를 AI로 분석하여 지식 그래프를 만들고, 하이브리드 검색(벡터 + 그래프)으로 질문에 답하는 GraphRAG 에이전트입니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/87f784de-25ba-4dcd-a3b4-0ae93824f9ed" />

---

## 🎯 프로젝트 개요

**하이브리드 검색 파이프라인**: 벡터 검색 → 그래프 확장 → Cypher 쿼리 → 결과 병합

1. **데이터 준비** - 위키피디아 스크래핑 → AI 개체/관계 추출 → JSON 저장
2. **그래프 생성** - JSON → Neo4j 지식그래프 + 벡터 임베딩 저장
3. **하이브리드 검색** - 자연어 질문 → 벡터 유사도 검색 → 그래프 순회 → Cypher 쿼리 생성 → 답변

---

## 📋 사전 요구사항

- **Python 3.12+**
- **OpenAI API Key** (또는 Ollama)
- **Neo4j Database** (Docker 권장)

---

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
uv sync
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Neo4j 실행 (Docker)

```bash
cd dockers
docker-compose up -d
```

### 4. 파이프라인 실행

```bash
# Step 1: 데이터 추출 (Wikipedia → JSON)
uv run 1_prepare_data_v3.py

# Step 2: 그래프 생성 (JSON → Neo4j + Embeddings)
uv run 2_ingest_data_v2.py

# Step 3: 하이브리드 검색 쿼리 (Interactive)
uv run 3_graphrag_agent_v3.py
```

---

## 📁 실행 파일 및 폴더

### 🔧 실행 스크립트 (버전별)

| 파일 | 주요 기능 | 출력 |
|------|----------|------|
| **`1_prepare_data_v3.py`** | Wikipedia 스크래핑 → OpenAI 개체/관계 추출 → 검증 | `output/raw_data_v3.json`, `knowledge_graph_v3.json` |
| **`2_ingest_data_v2.py`** | 지식그래프 Neo4j 저장 → 노드/관계 임베딩 생성 → 벡터 인덱스 구축 | Neo4j 데이터베이스 (`entity_embeddings`, `relationship_embeddings`) |
| **`3_graphrag_agent_v3.py`** | 하이브리드 검색: 벡터 유사도 → 그래프 확장 → Cypher 생성 → 답변 | 대화형 질의응답 시스템 |

> **버전 관리**: `_v1`, `_v2`, `_v3`은 기능 개선 버전. 최신 버전(`v3`) 사용 권장.

---

### 📂 주요 폴더

#### `output/`
**역할**: 데이터 파이프라인 중간 결과 저장

- `raw_data_v3.json` - Wikipedia 원본 데이터 (에피소드별 줄거리)
- `knowledge_graph_v3.json` - 추출된 노드/관계 JSON (Neo4j 입력용)
- `statistics_v3.json` - 데이터 통계 (노드 수, 관계 수, 처리 시간 등)

#### `dockers/`
**역할**: Neo4j + PostgreSQL 인프라 구성

- `docker-compose.yml` - Neo4j (Bolt 7687, HTTP 7474), PostgreSQL with pgvector
- `Dockerfile.fullstack` - 커스텀 환경 설정 (locale, timezone)
- `.env` - Docker 환경 변수 (비밀번호, 플러그인 등)

#### `config.py`
**역할**: 환경 변수 관리

- Neo4j 연결 정보 (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`)
- OpenAI/Ollama 모델 설정 (`OPENAI_MODEL`, `EMBEDDING_MODEL`)
- API 키 로딩 (`OPENAI_API_KEY`)

---

## 🔄 하이브리드 검색 파이프라인

### 1️⃣ 벡터 검색 (Vector Search)

- 사용자 질문을 임베딩 벡터로 변환
- Neo4j `entity_embeddings` 인덱스에서 코사인 유사도 기반 상위 K개 노드 검색
- 연관 관계에서 `relationship_embeddings` 인덱스로 상위 K개 관계 검색

### 2️⃣ 그래프 확장 (Graph Expansion)

- 벡터 검색으로 찾은 노드를 시작점으로 설정
- Neo4j에서 1~2 hop 그래프 순회하여 컨텍스트 확장
- 관련 에피소드, 캐릭터 간 관계 수집

### 3️⃣ Cypher 쿼리 생성 (Cypher Generation)

- LLM이 확장된 컨텍스트 + 스키마 정보를 분석
- Few-shot 예시 기반 Cypher 쿼리 자동 생성
- 구문 검증 및 재시도 메커니즘

### 4️⃣ 결과 병합 (Result Merging)

- Cypher 실행 결과를 자연어로 변환
- 에피소드 순서대로 정렬 및 맥락 추가
- 최종 답변 생성

---

## 🎮 예시 질문

```text
"카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?"
"토미오카 기유와 관계있는 모든 캐릭터를 알려줘."
"루이와 싸운 캐릭터는 누구야?"
"3번 이상 등장한 관계 타입은?"
```

---

## 📊 데이터 스키마

### 노드 타입
- **인간** - 귀살대원, 일반인 (properties: `name`, `embedding`)
- **도깨비** - 적 캐릭터 (properties: `name`, `embedding`)

### 관계 타입 (예시)
- `FIGHTS`, `PROTECTS`, `TRAINS`, `DEFEATS`, `RESCUES`, `BATTLES`, `JOINS`
- **공통 속성**: `episode_number`, `season`, `episode`, `context`, `embedding`

### 벡터 인덱스
- `entity_embeddings` - 노드 임베딩 (1024 dim, cosine)
- `relationship_embeddings` - 관계 임베딩 (1024 dim, cosine)

---

## 🛠️ 기술 스택

| 컴포넌트 | 기술 | 용도 |
|---------|------|------|
| 그래프 DB | Neo4j 5+ | 지식 그래프 저장, Cypher 쿼리 |
| 임베딩 | OpenAI `text-embedding-3-large` | 벡터 검색용 임베딩 생성 |
| LLM | OpenAI GPT-4o | 개체 추출, Cypher 생성, 답변 생성 |
| 오케스트레이션 | Python 3.12 | 파이프라인 실행 |
| 스크래핑 | BeautifulSoup | Wikipedia 데이터 수집 |

---

## 📚 참고 자료

- [귀멸의 칼날 시즌 1 Wikipedia](https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1)
- [Neo4j GraphRAG Library](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [OpenAI API 키 발급 가이드](https://github.com/dabidstudio/dabidstudio_guides/blob/main/get-openai-api-key.md)

---

## 🔍 검증 방법

### Neo4j 브라우저에서 확인

```cypher
// 전체 노드 수
MATCH (n) RETURN count(n)

// 관계 타입별 통계
MATCH ()-[r]-() RETURN type(r), count(r) ORDER BY count(r) DESC

// 벡터 인덱스 확인
SHOW INDEXES
```

### 하이브리드 검색 로그

`3_graphrag_agent_v3.py` 실행 시 콘솔에서 확인:
- 벡터 검색 결과 (상위 K개 유사 노드)
- 그래프 확장 컨텍스트
- 생성된 Cypher 쿼리
- 최종 답변

---

## ⚙️ 성능 최적화

- **캐싱**: 동일한 Wikipedia 페이지 재다운로드 방지 (`1_prepare_data_v3.py`)
- **배치 처리**: tqdm 진행률 표시로 대량 데이터 처리 추적
- **벡터 인덱스**: Neo4j HNSW 알고리즘으로 빠른 유사도 검색
- **Cypher 최적화**: `LIMIT` 절, 인덱스 활용 쿼리 생성



