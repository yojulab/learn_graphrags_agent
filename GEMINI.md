# GEMINI.md — GraphRAG Agent (Demon Slayer Season 1 Knowledge Graph) 하네스 헌장

> **Antigravity & Gemini 하네스 엔지니어링 시스템 가동 중**
> 본 프로젝트는 Claude Code의 `revfactory/harness` 및 `ralph-loop` 아키텍처를 Google Antigravity 및 Gemini 실행 환경에 맞춰 완벽히 재구성하고, **`docs/PRD.md` (Demon Slayer Season 1 Knowledge Graph PRD)**와 **듀얼 그래프 데이터베이스 백엔드(Neo4j 5+ 및 Apache AGE on PostgreSQL 16+)**, 그리고 **하이브리드 검색(Hybrid Retrieval) 에이전트 파이프라인**에 최적화된 하네스 규칙 체계를 준수합니다.

---

## 1. 하네스 시스템 트리거 및 라우팅 포인터

사용자의 요청 키워드에 따라 아래 스킬과 룰을 즉시 참조하여 작업을 수행합니다:

| 사용자 요청 키워드 | 대상 스킬 / 룰 | 설명 |
| :--- | :--- | :--- |
| **"하네스 구성해줘"**, **"하네스 설계"**, **"에이전트 팀 만들어줘"** | [.agents/skills/harness/SKILL.md](file:///apps/learn_graphrags_agent/.agents/skills/harness/SKILL.md) | 프로젝트 도메인 분석 후 최적의 에이전트 팀과 스킬 설계 |
| **"하네스 회고"**, **"하네스 피드백 반영"**, **"하네스 진화"** | [.agents/skills/harness-evolve/SKILL.md](file:///apps/learn_graphrags_agent/.agents/skills/harness-evolve/SKILL.md) | 실행 델타 포착, 원리 일반화, 룰 및 에이전트 지속 개선 |
| **"루프 실행"**, **"ralph loop"**, **"끝날 때까지 반복"**, **"자율 루프"** | [.agents/skills/ralph-loop/SKILL.md](file:///apps/learn_graphrags_agent/.agents/skills/ralph-loop/SKILL.md) | Ralph Loop 자율 반복 실행, 완료 약속 및 서킷 브레이커 관리 |
| **GraphRAG 도메인 & 파이프라인 규격** | [.agents/rules/graphrag-domain.md](file:///apps/learn_graphrags_agent/.agents/rules/graphrag-domain.md) | 귀멸의 칼날 지식그래프, 듀얼 백엔드(Neo4j & Apache AGE), v1~v4 파이프라인, psql 검증 |
| **지식그래프 스키마 & 10대 골든 룰** | [.agents/rules/graphrag-rules.md](file:///apps/learn_graphrags_agent/.agents/rules/graphrag-rules.md) | 노드(인간/도깨비), 24개 관계, 4대 질문 유형, Read-Only 쿼리, 생각태그 제거 등 10대 수칙 |
| **통합 에이전트 하네스 프롬프트 규격** | [docs/agent/harness_prompt.md](file:///apps/learn_graphrags_agent/docs/agent/harness_prompt.md) | 질문 의도 분류기, Text2Cypher, Apache AGE openCypher SQL, 컨텍스트 포맷터, 답변 합성기 |
| **에이전트 시스템 프롬프트 규격** | [docs/agent/system_prompt.md](file:///apps/learn_graphrags_agent/docs/agent/system_prompt.md) | 런타임 시스템 프롬프트 지침, 읽기 전용 가드레일, 시퀀스 다이어그램 |
| **UI 태깅 & 인스펙션 규격** | [.agents/rules/ui-tagging-inspection.md](file:///apps/learn_graphrags_agent/.agents/rules/ui-tagging-inspection.md) | FastAPI Web UI 일련번호 식별자 체계, `Alt+I` 단축키 및 4대 정보 원클릭 복사 |
| **Git 커밋 규약 & 업무 종료 수칙** | [.agents/rules/git-workflow.md](file:///apps/learn_graphrags_agent/.agents/rules/git-workflow.md) | 업무 종료 시 Git Commit 필수화, 원자적 커밋, 메시지 컨벤션 |
| **하네스 아키텍처 & 품질 원칙** | [.agents/rules/harness-engineering.md](file:///apps/learn_graphrags_agent/.agents/rules/harness-engineering.md) | 6대 팀 패턴, 5대 품질 검증, 점진적 공개 원칙 |
| **루프 엔지니어링 원칙 & 서킷 브레이커** | [.agents/rules/loop-engineering.md](file:///apps/learn_graphrags_agent/.agents/rules/loop-engineering.md) | Iteration > Perfection, 상태 디스크 영속화, 서킷 브레이커 규정 |

---

## 2. GraphRAG Agent 핵심 업무 지시 원칙 및 개발 가드레일

1. **`docs/PRD.md` 기준 4단계 파이프라인 엄격 준수**:
   - **v1 (Baseline)**: 기본 Text2Cypher 및 Neo4j 질의 검증.
   - **v2 (Metadata Filter)**: 속성 기반 필터링 및 벡터 RAG 결합.
   - **v3 (Production Hybrid)**: HNSW 벡터 인덱스 + 1~2 hop 그래프 확장 + Neo4j 네이티브 Cypher.
   - **v4 (Apache AGE Multi-Model)**: PostgreSQL 16+ & Apache AGE 1.5+, `ag_catalog`, openCypher in SQL(`cypher(...)`), 엔터프라이즈 RDBMS 하이브리드 결합.
2. **듀얼 백엔드 격리 및 동등성 보장**:
   - **Neo4j 5+**: Bolt 프로토콜(`7687`), HTTP(`7474`), HNSW 인덱스(`entity_embeddings_*`, `rel_embeddings_*`).
   - **PostgreSQL 16+ & Apache AGE**: 포트(`5432`), `ag_catalog`, `demon_slayer` 그래프, `SELECT * FROM cypher(...) as (...)`.
   - 동일한 정제 데이터(`output/knowledge_graph_v*.json`)를 기반으로 두 백엔드가 동등한 질의 품질을 제공해야 합니다.
3. **GraphRAG 10대 엔지니어링 골든 룰 절대 준수**:
   - 읽기 전용 쿼리 안전성(Read-Only Safety Guardrails: `MATCH`/`OPTIONAL MATCH` 전용, 변형 쿼리 차단).
   - 엄격한 스키마 바인딩(Strict Schema Grounding: 라벨 `인간`/`도깨비`, 24개 관계).
   - 생각 태그 정제(Think Tag Sanitization: `<think>...</think>` 제거).
   - 데이터 적재 멱등성 보장(Idempotent Graph Ingestion: 재실행 시 초기화 및 인덱스 재구축).
   - 1024차원 고품질 임베딩(`bge-m3:567m` 코사인 유사도 검색).
   - 1~2 Hop 동적 그래프 확장(단순 키워드 극복 및 심층 문맥 확보).
   - 환각 방지 및 미확인 사실 명시("지식 그래프 데이터상 확인되지 않는 정보입니다").
   - 중앙 설정 파일 단일 원천(`.env` & `config.py`).
   - 원자적 Git Commit 및 무한 폴링 차단.
4. **알맞은 업무 종료 시 Git Commit 의무화**:
   - 모든 에이전트는 기획/아키텍처 문서 생성, 룰 갱신, 파이프라인 스크립트 작성, 검증 완료 등 유의미한 작업 단위를 완수했을 때 **반드시 원자적 Git Commit을 수행**합니다.
   - 커밋 전 `.env` 등 민감 정보 누출 방지 및 `.gitignore` 적용 여부를 엄격히 확인합니다.

---

## 3. GraphRAG 하네스 전담 에이전트 팀 (Roles)

| 에이전트 롤 | 모델 티어 | 주 임무 및 전담 산출물 |
| :--- | :--- | :--- |
| **GraphRAG System Architect** | Gemini Pro | 전체 4단계(v1~v4) 파이프라인 설계, 듀얼 DB (Neo4j & Apache AGE) 아키텍처 및 Docker 인프라 구성, `.env` 및 `config.py` 중앙 관리, 시스템 명세서 총괄 |
| **Knowledge Extraction & Data Specialist** | Gemini Flash/Pro | 위키피디아 시즌 1 스크래핑(Playwright/BS4), Pydantic v2 데이터 정규화, 한국어 인명/도깨비명 표준화, 24개 관계 및 노드 라벨 추출, `output/` JSON 아티팩트 관리 |
| **Neo4j Native Graph Specialist** | Gemini Pro | Neo4j 5+ Bolt 연동, 노드/관계 1024차원 `bge-m3` 벡터 적재, HNSW 인덱스 자동 구축, Cypher 쿼리 튜닝 및 1~2 hop 그래프 확장 최적화 |
| **Apache AGE & Relational Specialist** | Gemini Pro | PostgreSQL 16+ & Apache AGE 1.5+ 연동, `LOAD 'age';`, `create_graph`, 정점/간선 라벨 등록, `agtype` 데이터 파싱, SQL-openCypher 하이브리드 결합 질의 최적화 |
| **Hybrid Retrieval & Agent Prompt Specialist** | Gemini Pro | 4단계 하네스 프롬프트(질문 의도 분류, Text2Cypher / Text2openCypher 생성, 컨텍스트 포맷터, 답변 합성기) 튜닝, `<think>...</think>` 태그 정제, 한국어 개조식 답변 작성 |
| **Quality & Playwright E2E Assurer** | Gemini Pro | CLI 파이프라인 검증, 인덱스 정합성(`check_indexes.py`), Direct psql 쿼리 검증, FastAPI 웹 UI Playwright E2E 테스트, Hallucination 차단 및 회귀 검증 |

---

## 4. 하네스 변경 이력 (Harness Changelog)

| 일자 | 변경 내용 | 대상 | 사유 |
| :--- | :--- | :--- | :--- |
| **2026-10-17** | `docs/PRD.md` 기준 GraphRAG 맞춤형 하네스 전면 전환 | 전체 하네스 시스템 | `docs/PRD.md` (Demon Slayer Season 1 Knowledge Graph PRD) 사양 반영 및 WorkshopSync 잔재 전면 개편 |
| **2026-10-17** | Neo4j & Apache AGE 듀얼 백엔드 하네스 규칙 수립 | `GEMINI.md`, `AGENTS.md`, `graphrag-domain.md` | Neo4j (v1~v3) 및 PostgreSQL 16+ Apache AGE (v4) 멀티모델 파이프라인 규격 탑재 |
| **2026-10-17** | GraphRAG 10대 엔지니어링 골든 룰 및 스키마 바인딩 탑재 | `graphrag-rules.md`, `plugin.json` | 읽기 전용 안전성, 스키마 바인딩(노드 2종, 관계 24종), 생각 태그 정제, 환각 방지 등 10대 골든 룰 정립 |
| **2026-10-17** | 듀얼 백엔드 4단계 하네스 프롬프트 및 시스템 프롬프트 규격화 | `docs/agent/harness_prompt.md`, `docs/agent/system_prompt.md` | 질문 의도 분류기, Neo4j Cypher 생성기, Apache AGE openCypher SQL 래핑 생성기, 컨텍스트 포맷터, 한국어 개조식 답변 합성기 완비 |
| **2026-10-17** | Git 워크플로우, 루프 엔지니어링, UI 인스펙션 GraphRAG 환경 정합화 | `.agents/rules/*` | `/apps/learn_graphrags_agent` 절대경로 일치화 및 GraphRAG FastAPI 웹/CLI 검증 기준 반영 |
