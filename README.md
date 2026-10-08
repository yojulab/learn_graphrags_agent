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

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
uv sync
```

### 2. 환경 변수 설정 (`.env`)

프로젝트 루트의 `.env` 파일에서 전체 시스템 및 모델 설정을 관리합니다:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=admin123
NEO4J_DATABASE=db-graphrag-agent

LLM_MODEL=sam860/exaone-4.0:1.2b-thinking-Q8_0
EMBEDDING_MODEL=bge-m3:567m
MODEL_API_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

### 3. Docker 인프라 실행

```bash
cd dockers
docker-compose up -d
cd ..
```

### 4. 파이프라인 실행 (최신 v3 버전)

```bash
# Step 1: 데이터 추출 (Wikipedia → JSON)
uv run src/v3/1_prepare_data_v3.py

# Step 2: 그래프 생성 (JSON → Neo4j + Vector Embeddings)
uv run src/v3/2_ingest_data_v3.py

# Step 3: 하이브리드 검색 쿼리 (Interactive)
uv run src/v3/3_graphrag_agent_v3.py

# Utility: 벡터 인덱스 상태 확인
uv run src/utils/check_indexes.py
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
│       └── harness_prompt.md         # 에이전트 하네스 프롬프트 규격서
└── src/                              # 버전별 실행 코드 폴더
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
        └── check_indexes.py
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
