# Git 워크플로우 & 업무 종료 커밋 규약 (Git Commit Rules for GraphRAG Agent)

본 문서는 `learn_graphrags_agent` 프로젝트에서 모든 에이전트가 작업을 수행하고 종료할 때 준수해야 하는 **Git Commit 표준 지침**이다.

---

## 1. 커밋 시점 및 의무 원칙

### 1-1. 업무 종료 시 Git Commit 필수
- 에이전트는 PRD/명세서 작성, 파이프라인 스크립트(v1~v4) 개선, 스키마 및 프롬프트 정의, 룰 수정, 테스트 작성 등 **유의미한 업무 단위를 마칠 때마다 반드시 Git Commit을 수행**해야 한다.
- Ralph Loop 실행 중에도 각 마일스톤이나 서브태스크가 완료되면 즉시 커밋하여 디스크 상에 상태를 영속화(State on Disk)한다.

### 1-2. 원자적 커밋 (Atomic Commits)
- 여러 성격의 작업을 한데 뭉뚱그려 커밋하지 않고, 독립적인 기능/문서 단위별로 원자적으로 커밋한다.
- 커밋 전 `git status`와 `git diff`를 확인하여 의도하지 않은 파일이 포함되지 않았는지 점검한다.

---

## 2. 커밋 메시지 컨벤션

커밋 메시지는 다음 접두어를 사용하여 명확히 기술한다:

| 타입 | 의미 | 예시 |
| :--- | :--- | :--- |
| **`arch:`** | 시스템/데이터베이스/백엔드 아키텍처 설계 | `arch: PostgreSQL Apache AGE v4 멀티모델 파이프라인 설계` |
| **`docs:`** | PRD, 하네스 프롬프트 명세, 가이드라인 작성 | `docs: docs/agent/harness_prompt.md 듀얼 백엔드 규격 갱신` |
| **`feat:`** | 파이프라인 스크립트, 에이전트, 프롬프트 템플릿 추가 | `feat: src/v4/3_graphrag_agent_v4.py Apache AGE 에이전트 구현` |
| **`rule:`** | 하네스 규칙, 에이전트 역할, 스킬 변경 | `rule: GraphRAG 도메인 규칙 및 10대 엔지니어링 골든 룰 갱신` |
| **`test:`** | Playwright E2E 검증, 인덱스 점검, 쿼리 테스트 추가 | `test: src/utils/check_indexes.py Neo4j HNSW 인덱스 검증 추가` |
| **`chore:`** | .gitignore, 환경변수 템플릿, 패키지 설정, uv 설정 | `chore: .env 및 config.py Apache AGE 설정 변수 정비` |

---

## 3. 커밋 전 보안 및 무결성 체크리스트

1. **`.env` 파일 배제**: 실제 비밀키나 API 토큰이 담긴 로컬 `.env` 파일이 포함되지 않았는가? (`.gitignore` 적용 여부 확인)
2. **스키마 바인딩 확인**: 코드/프롬프트 내 노드 라벨(`인간`, `도깨비`) 및 24개 관계 타입이 표준 스키마와 일치하는가?
3. **읽기 전용 안전성 확인**: 생성된 Cypher 쿼리 템플릿에 데이터 변형 구문(`CREATE`, `DELETE` 등)이 배제되었는가?
4. **완료 약속 연계**: Ralph Loop 상태 파일(`.agents/loop/state.json`)의 진전 사항과 커밋 내용이 일치하는가?
