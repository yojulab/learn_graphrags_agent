# 하네스 엔지니어링 헌장 (Harness Engineering Charter for GraphRAG Agent)

이 문서는 Google Antigravity / Gemini 환경에서 `learn_graphrags_agent` 프로젝트의 에이전트 팀과 스킬 체계를 구축하고 운영할 때 준수해야 하는 엔지니어링 표준 및 행동 헌장이다.

---

## 1. 하네스(Harness)의 철학과 목적

### 1-1. 단일 프롬프트의 한계와 하네스의 필요성
- 복잡한 지식그래프 구축(NER, 관계 추출, Cypher 생성, 임베딩, 다중 홉 탐색)을 단일 프롬프트로 처리할 경우 **주의력 분산(Attention Dilution)**, **스키마 환각(Hallucination)**, **문법 오류**가 발생한다.
- **하네스(Harness)**는 AI 에이전트가 복잡한 지식그래프 엔지니어링 작업을 수행할 수 있도록 감싸는 **신뢰 레이어(Trust Layer)**이자 **시스템 인프라**다.
- 작업의 성격에 따라 역할을 세분화하고, 에이전트에게는 **"누가 일하는가(Who/Role)"**를, 스킬에는 **"그 일을 어떻게 하는가(How/Procedure)"**를 명확히 분리한다.

### 1-2. 핵심 목표
1. **결정론적 품질 보장**: "그럴듯하지만 문법/스키마가 틀린 Cypher 쿼리"를 시스템 수준의 파서와 검증 루프로 걸러낸다.
2. **효율적 컨텍스트 활용**: 점진적 공개(Progressive Disclosure)를 통해 필요한 순간에만 스키마와 Few-shot 예시를 로드한다.
3. **지속적 진화(Evolution)**: 실행 과정의 실패(Cypher 실행 에러, 누락된 엔티티 등)를 포착하여 시스템 프롬프트 및 Few-shot에 즉각 환류한다.
4. **원자적 상태 영속화 (Git Commit)**: 파이프라인 스크립트 작성, 프롬프트 개선, 인덱스 생성 등 유의미한 작업 단위 완료 시 반드시 Git Commit을 수행한다.
5. **듀얼 백엔드 격리 및 동등성 보장**: Neo4j(Bolt, 7687)와 PostgreSQL Apache AGE(5432)의 인프라와 쿼리 문법을 독립적으로 제어하면서 동일한 지식 무결성을 유지한다.

---

## 2. Gemini 아키텍처 패턴 (6대 기본 패턴)

| 패턴 | 구조 | 제어 흐름 특성 | Antigravity 최적 실행 모드 |
| :--- | :--- | :--- | :--- |
| **1. 파이프라인 (Pipeline)** | `A → B → C → D` | 순차 의존적, 단계별 산출물이 다음 단계의 입력이 됨 (데이터수집 → 적재/임베딩 → 에이전트) | 모드 A: 순차 백그라운드 태스크 및 스크립트 실행 |
| **2. 분산·통합 (Fan-out / Fan-in)** | `입력 → {A, B, C} → 통합` | 독립 작업의 병렬 처리 (에피소드별 병렬 추출 후 지식그래프 병합) | 모드 A: 병렬 실행 후 최종 취합 |
| **3. 전문가 풀 (Expert Pool)** | `분류기 → 적합 전문가 선택` | 질문 유형(`SINGLE_ENTITY`, `RELATIONSHIP` 등)에 따른 동적 라우팅 | 온디맨드 스킬 로드 및 전문가 프롬프트 호출 |
| **4. 생성·검증 (Producer-Reviewer)** | `생성 ⇄ 검증 (피드백 루프)` | Cypher 생성기 ⇄ Read-Only 쿼리 파서 및 문법 검증기 반복 루프 | 적대적 검증자 분리 및 교차 검토 |
| **5. 감독자 (Supervisor)** | `감독자 ⇄ 작업자 풀 (동적 배분)` | 동적 작업 큐, 상태 추적 및 재할당 | 지속 세션 오케스트레이터 및 작업 관리 |
| **6. 계층형 위임 (Hierarchical)** | `상위 에이전트 → 하위 팀 (최대 2단계)` | 복합 프로젝트의 하향식 재귀 분할 | 도메인별 리더 에이전트와 서브태스크 분리 |

---

## 3. 5대 품질 검증 패턴 (Workflow-Native Quality Patterns)

1. **적대적 검증 (Adversarial Verification)**: Cypher 쿼리 내 `CREATE`, `DELETE` 등 위험 키워드 감지 및 차단 검증.
2. **심사위원단 (Judge Panels)**: 텍스트2Cypher 생성 시 2개 이상의 쿼리 후보를 생성하여 스키마 정합성과 비용을 비교 심사.
3. **Loop-until-dry (발견 0건 수렴 루프)**: 지식그래프 적재 시 결함(미등록 라벨, 끊어진 간선) 0건 수렴까지 반복.
4. **다각 스윕 (Multi-angle Sweeps)**: 벡터 검색 유사도, 그래프 탐색 깊이(1~2 hop), LLM 추론 태그 제거 여부 다각도 점검.
5. **누락 검토자 (Completeness Critic)**: PRD.md 요구사항(노드 라벨 2종, 관계 24종, 4개 질의 유형, 듀얼 백엔드) 대비 누락 요소만을 집중 검토.

---

## 4. 업무 지시 및 개발 단계 가드레일

- 모든 에이전트는 `docs/PRD.md` 및 `docs/agent/harness_prompt.md` 사양을 엄격히 준수한다.
- 모든 에이전트는 각 단위 마일스톤 또는 세부 작업 완료 시 [.agents/rules/git-workflow.md](file:///apps/learn_graphrags_agent/.agents/rules/git-workflow.md)에 따라 **Git Commit을 반드시 수행**한다.
