# GraphRAG 도메인 용어 체계 및 10대 엔지니어링 골든 룰
## (GraphRAG Terminology & 10 Golden Engineering Rules)

- **참조 문서**: `docs/PRD.md`, `docs/agent/harness_prompt.md`, `docs/agent/system_prompt.md`
- **적용 대상**: `learn_graphrags_agent` 파이프라인을 설계, 구현, 검증하는 모든 하네스 에이전트 및 개발자
- **핵심 목표**: 비정형 텍스트로부터 지식그래프를 안정적으로 구축하고, LLM 환각(Hallucination) 없이 정확한 지식 기반 질의응답을 수행하도록 시스템 수준의 안전망을 제공함.

---

## 1. 지식그래프 핵심 도메인 스키마 및 용어 체계 (Glossary)

### 1-1. 정점 라벨 (Node Labels)
지식그래프의 모든 정점은 아래 2개 표준 라벨 중 하나를 반드시 부여받습니다:
- **`인간`**: 귀살대원(카마도 탄지로, 토미오카 기유 등), 주(柱), 가족 및 일반인 캐릭터.
- **`도깨비`**: 키부츠지 무잔, 십이지귀월(루이 등) 및 혈귀 캐릭터.
- **공통 프로퍼티**: `name` (STRING, 유일 식별명), `role` (주인공, 귀살대 수주, 십이지귀월 등), `first_appearance` (에피소드 번호).

### 1-2. 간선 관계 유형 (Relationship Types - 24개 표준 관계)
모든 간선은 다음 24개의 명확한 방향성 관계 동사만을 사용합니다:
```text
FIGHTS, PROTECTS, TRAINS, TRAINS_WITH, KNOWS, FAMILY_OF, SIBLING_OF, ALLY_OF, 
ENEMY_OF, DEFEATS, SAVES, RESCUES, MEETS, ENCOUNTERS, GUIDES, ATTACKS, 
DEFENDS, TRANSFORMS, JOINS, SUPPORTS, REUNITES_WITH, BATTLES, HEALS, TEACHES
```
- **간선 프로퍼티**: `episode_number` (INTEGER), `season` (INTEGER, 기본값 1), `context` (STRING, 관계 발생 맥락 요약).

### 1-3. 질문 분류 체계 (Query Intent Types - 4대 유형)
자연어 질문은 Stage 1 질문 분석기를 통해 아래 4가지 중 하나로 엄격히 분류됩니다:
1. **`SINGLE_ENTITY`**: 특정 캐릭터나 도깨비 1인의 활약상, 행적, 배경 질의.
2. **`RELATIONSHIP`**: 두 명 이상의 캐릭터/도깨비 간의 대결, 구출, 동맹, 가족 관계 질의.
3. **`EPISODE_SPECIFIC`**: 특정 에피소드 번호(예: 1화, 19화)에 발생한 사건 및 등장인물 질의.
4. **`GENERAL`**: 전체 인물 통계, 도깨비 목록, 시즌 전반의 개괄적 질의.

---

## 2. GraphRAG 10대 엔지니어링 골든 룰 (10 Golden Rules)

1. **[읽기 전용 쿼리 안전성 원칙 (Read-Only Safety Guardrails)]**:
   생성되는 모든 Cypher 및 openCypher 쿼리는 `MATCH` 및 `OPTIONAL MATCH` 구문만 사용해야 한다. `CREATE`, `MERGE`, `SET`, `DELETE`, `DROP` 등 그래프 데이터를 변경하는 돌연변이 구문은 일절 금지되며 정규식으로 사전 차단한다.
2. **[엄격한 스키마 바인딩 (Strict Schema Grounding)]**:
   LLM은 임의의 노드 라벨이나 관계 타입을 지어낼 수 없다. 오직 `config.py`에 정의된 `NODE_LABELS` (`인간`, `도깨비`)와 `RELATIONSHIP_TYPES` (24종)만을 기반으로 쿼리를 생성해야 한다.
3. **[생각 태그 정제 원칙 (Think Tag Sanitization)]**:
   EXAONE 4.0 등 추론 모델이 생성하는 `<think>...</think>` 태그는 쿼리 실행 및 최종 사용자 응답 전 반드시 정규식(`re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)`)으로 완전 제거해야 한다.
4. **[데이터 적재 멱등성 보장 (Idempotent Graph Ingestion)]**:
   데이터 적재 스크립트(`2_ingest_data_v3.py`, `2_ingest_data_v4.py`)는 재실행 시 기존 데이터를 깨끗이 초기화(`drop_graph` / `DETACH DELETE`)하고 인덱스를 재구축하여 데이터 중복 및 상태 오염을 원천 방지한다.
5. **[듀얼 백엔드 일관성 유지 (Dual-Backend Parity)]**:
   v3(Neo4j)와 v4(Apache AGE)는 동일한 정규화 데이터(`output/knowledge_graph_v*.json`)를 입력받아 동등한 노드/관계 토폴로지와 질의 결과를 제공해야 한다.
6. **[1024차원 고품질 벡터 임베딩 (1024-dim Vector Embedding)]**:
   노드와 관계의 의미적 탐색을 위해 `bge-m3:567m` 기반 1024차원 벡터를 생성하며, 코사인 유사도 검색 인덱스(Neo4j HNSW, Postgres 벡터 테이블)를 필수로 유지한다.
7. **[1~2 Hop 동적 그래프 확장 (Dynamic Graph Expansion)]**:
   벡터 검색으로 도출된 시드 엔티티로부터 1~2 hop 깊이로 그래프를 확장 순회하여 단순 키워드 매칭을 넘어선 심층적 문맥을 하이브리드 결합한다.
8. **[환각 방지 및 미확인 명시 원칙 (Anti-Hallucination Fallback)]**:
   그래프 데이터베이스 검색 결과에 없는 사실은 절대 추측하여 답변하지 않는다. 검색 결과가 부재할 경우 `"지식 그래프 데이터상 확인되지 않는 정보입니다"`라고 명확히 안내한다.
9. **[중앙 설정 파일 단일 원천 (Single Source of Truth)]**:
   모든 모델명, DB 접속 정보, 임베딩 차원, 라벨/관계 목록은 루트 `.env`와 `config.py`에 집중 관리하며, 하드코딩을 절대 금지한다.
10. **[원자적 커밋 및 무한 폴링 차단 (Atomic Commit & Zero Polling)]**:
    의미 있는 파이프라인 구현, 스키마 갱신, 프롬프트 개선 완료 시 원자적 Git Commit을 수행한다. 백그라운드 태스크 대기 시 무한 폴링 루프를 돌지 않고 반응형 웨이크업에 위임한다.
