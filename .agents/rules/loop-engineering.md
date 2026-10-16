# 루프 엔지니어링 헌장 (Loop Engineering Charter for GraphRAG Agent)

이 문서는 Ralph Loop(Ralph Wiggum Technique)을 Google Antigravity / Gemini 환경에 맞춰 체계화한 **루프 엔지니어링(Loop Engineering)** 표준 규약이다.

---

## 1. 루프 엔지니어링의 철학과 핵심 원칙

### 1-1. 철학: 반복이 완벽을 이긴다 (Iteration > Perfection)
- 단일 프롬프트에서 완벽한 결과를 요구하면 모델은 과도한 추론 부담과 컨텍스트 한계로 실패하기 쉽다.
- 루프 엔지니어링은 **"실패를 정상적인 진단 데이터(Failures as Data)"**로 수용하며, 목표에 도달할 때까지 자율적이고 집요하게 수정-검증-재시도를 반복한다.

### 1-2. 5대 핵심 기둥 (5 Pillars of Loop Engineering)

1. **상태의 디스크 영속화 (State on Disk)**:
   - 대화 세션의 단기 메모리에 의존하지 않는다.
   - 현재 진행 상태, 남은 태스크 목록, 완료된 항목, 테스트 결과는 항상 **디스크의 파일(`task.md`, `state.json`, Git 커밋)**에 기록한다.
2. **명확한 완료 약속 (Completion Promise)**:
   - 루프 종료 조건은 주관적 판단이 아닌 기계적으로 검증 가능한 문자열 또는 테스트 결과여야 한다.
   - 예: `<promise>COMPLETED</promise>`, `ALL_INDEXES_VALIDATED`, 또는 종료 코드 0.
3. **신선한 컨텍스트 vs 세션 유지 (Fresh Context vs In-Session)**:
   - **세션 유지형 (In-Session Stop Hook)**: 최근 작업의 세부 맥락을 유지한 채 빠른 수정을 이어갈 때 적합.
   - **신선한 컨텍스트형 (Fresh Context Loop)**: 장기 작업 시 컨텍스트 오염(Context Rot)을 방지하기 위해 매 반복마다 디스크의 상태 파일만 읽고 새로 시작.
4. **회로 차단기 (Circuit Breakers & Safety Ceilings)**:
   - 무한 루프 및 리소스 낭비를 차단하기 위해 **최대 반복 횟수(Max Iterations)**, **무진전 감지(No-Progress Detection)**를 필수 적용.
5. **원자적 단위 전진 (Atomic Forward Progress)**:
   - 한 번의 루프 턴에서는 전체를 다 바꾸려 하지 않고, 하나의 서브태스크를 구현하고 검증하는 단위 작업을 수행한다.

---

## 2. Gemini 네이티브 루프 구현 방식 3종

| 모드 | 메커니즘 | 최적 사용 시점 |
| :--- | :--- | :--- |
| **모드 1: Stop Hook 인터셉트** | `.agents/hooks.json`의 `Stop` 이벤트에서 `decision: "continue"` 반환 | 단일 작업의 스크립트 실행/테스트 에러 수렴, TDD 루프 |
| **모드 2: Fresh Context 셸 스크립트** | `scripts/ralph_loop.sh`를 통한 외부 `while` 루프 및 디스크 상태 갱신 | 대규모 데이터 추출, 에피소드 전수 파이프라인 정제 |
| **모드 3: Antigravity `/goal` & 스케줄** | IDE 내장 `/goal` 슬래시 커맨드 및 `schedule` 백그라운드 크론 | 인터랙티브 세션 중 장기 실행 자율 완료 |

---

## 3. 루프 실행 시 에이전트 행동 수칙

1. **루프 시작 전 체크포인트 생성**:
   - 루프에 진입하기 전 반드시 Git 변경 사항을 스테이징하거나 임시 커밋을 생성한다.
2. **매 턴마다 검증 실행**:
   - 코드를 수정한 후에는 반드시 테스트나 린트 스크립트를 실행하여 진전 여부를 확인한다 (`uv run src/utils/check_indexes.py` 등).
3. **완료 약속 출력 엄수**:
   - 설정된 모든 검증 조건을 통과했을 때에만 지정된 완료 약속 토큰(예: `PROMISE_MET: ALL_TESTS_PASSED`)을 출력하고 루프를 탈출한다.
4. **정체 시 전략 수정**:
   - 2회 연속 동일한 에러가 발생하면 동일한 코드를 반복하지 않고, 가설을 바꾸거나 접근법을 전면 재검토한다.
