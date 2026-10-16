# AGENTS.md — 에이전트 팀 행동 강령 및 표준 지침
## (GraphRAG Agent: Demon Slayer Season 1 Knowledge Graph)

본 워크스페이스는 [GEMINI.md](file:///apps/learn_graphrags_agent/GEMINI.md), [.agents/rules/graphrag-domain.md](file:///apps/learn_graphrags_agent/.agents/rules/graphrag-domain.md), [.agents/rules/graphrag-rules.md](file:///apps/learn_graphrags_agent/.agents/rules/graphrag-rules.md), [.agents/rules/harness-engineering.md](file:///apps/learn_graphrags_agent/.agents/rules/harness-engineering.md), [.agents/rules/git-workflow.md](file:///apps/learn_graphrags_agent/.agents/rules/git-workflow.md), [.agents/rules/loop-engineering.md](file:///apps/learn_graphrags_agent/.agents/rules/loop-engineering.md), [.agents/rules/ui-tagging-inspection.md](file:///apps/learn_graphrags_agent/.agents/rules/ui-tagging-inspection.md)에 정의된 하네스 엔지니어링 체계를 따릅니다.

---

## 1. 에이전트 필수 준수 사항

1. **상시 룰 참조**:
   - 작업을 시작하기 전 반드시 [GEMINI.md](file:///apps/learn_graphrags_agent/GEMINI.md), 도메인 가이드([.agents/rules/graphrag-domain.md](file:///apps/learn_graphrags_agent/.agents/rules/graphrag-domain.md)), 10대 골든 룰([.agents/rules/graphrag-rules.md](file:///apps/learn_graphrags_agent/.agents/rules/graphrag-rules.md)), 하네스 엔지니어링 헌장([.agents/rules/harness-engineering.md](file:///apps/learn_graphrags_agent/.agents/rules/harness-engineering.md)), Git 워크플로우 룰([.agents/rules/git-workflow.md](file:///apps/learn_graphrags_agent/.agents/rules/git-workflow.md))을 준수한다.
2. **듀얼 그래프 DB 백엔드 규격 엄수 (Neo4j & Apache AGE)**:
   - **v1 ~ v3**: Neo4j 5+ (Bolt protocol `7687`, HTTP `7474`, HNSW 벡터 인덱스 `entity_embeddings`, `rel_embeddings`).
   - **v4**: PostgreSQL 16+ & Apache AGE 1.5+ (포트 `5432`, `ag_catalog`, `demon_slayer` 그래프, openCypher in SQL `SELECT * FROM cypher(...) as (...)`).
   - 두 백엔드는 동등한 데이터 토폴로지와 일관된 하이브리드 검색 품질을 제공해야 한다.
3. **GraphRAG 10대 엔지니어링 골든 룰 엄수**:
   - **읽기 전용 안전성 (Read-Only Guardrails)**: 모든 생성 쿼리는 `MATCH` / `OPTIONAL MATCH`만 허용하며, 데이터 변형 구문(`CREATE`, `MERGE`, `SET`, `DELETE`, `DROP`)은 절대 금지.
   - **엄격한 스키마 바인딩 (Strict Schema Grounding)**: 정점 라벨(`인간`, `도깨비`) 2종과 24개 표준 관계 동사만 사용.
   - **생각 태그 정제 (Think Tag Sanitization)**: 추론 모델(`sam860/exaone-4.0:1.2b-thinking-Q8_0`) 응답 내 `<think>...</think>` 블록을 정규식으로 사전 제거.
   - **적재 멱등성 (Idempotency)**: DB 적재 스크립트는 재실행 시 기존 데이터를 초기화하여 중복과 오염을 방지.
   - **1024차원 임베딩**: `bge-m3:567m` 기반 코사인 유사도 벡터 검색 인덱스 유지.
   - **1~2 Hop 그래프 확장**: 시드 노드로부터 1~2 hop 동적 순회로 다각도 문맥 확보.
   - **환각 방지 (Anti-Hallucination)**: 데이터 부재 시 `"지식 그래프 데이터상 확인되지 않는 정보입니다"`로 명확히 응답.
   - **중앙 설정 엄수**: `.env` 및 `config.py`를 단일 진실 공급원(SSOT)으로 사용.
   - **원자적 Git 커밋**: 의미 있는 작업 완료 시 즉시 커밋.
   - **무한 폴링 금지**: 백그라운드 태스크 대기 시 반응형 웨이크업에 위임.
4. **4단계 하네스 프롬프트 파이프라인 준수**:
   - `docs/agent/harness_prompt.md`에 명시된 4단계(질문 분석기 ➔ 시드 벡터 추출 ➔ 백엔드별 Text2Cypher 생성기 ➔ 컨텍스트 포맷터 ➔ 최종 답변 합성기)의 입출력 규격을 준수한다.
5. **알맞은 업무 종료 시 Git Commit 필수화**:
   - 모든 에이전트는 기획/아키텍처 문서 생성, 스크립트 작성, 스펙 갱신, 검증 완료 등 유의미한 작업 단위 완료 시 **반드시 원자적 Git Commit을 수행**한다.
   - 커밋 메시지 컨벤션(`arch:`, `docs:`, `feat:`, `rule:`, `test:`, `chore:`)을 준수하며, `.env` 등 민감 파일이 스테이징되지 않도록 사전 점검한다.
6. **역할과 스킬의 분리**:
   - 에이전트는 본인의 정의된 책임(Who) 범위를 벗어나지 않는다.
   - 구체적인 실행 절차(How)는 `.agents/skills/`에 정의된 스킬 가이드를 따른다.
7. **루프 엔지니어링과 상태 영속화**:
   - 반복 루프 작업 수행 시 세션 메모리에 의존하지 않고, 항상 디스크의 상태 파일([.agents/loop/state.json](file:///apps/learn_graphrags_agent/.agents/loop/state.json), [.agents/loop/task.md](file:///apps/learn_graphrags_agent/.agents/loop/task.md))과 Git 커밋에 진행 상황을 원자적으로 기록한다.
   - 사전에 정의된 검증 기준과 완료 약속(Completion Promise)이 기계적으로 입증되었을 때에만 약속 토큰을 출력하고 종료한다.
8. **자가 진화 참여**:
   - 작업 완료 후 오류 패턴이나 사용자 피드백이 발생하면 [harness-evolve](file:///apps/learn_graphrags_agent/.agents/skills/harness-evolve/SKILL.md) 스킬을 통해 시스템 지식으로 일반화하여 환류한다.
