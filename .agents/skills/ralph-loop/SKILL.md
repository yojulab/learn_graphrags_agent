---
name: ralph-loop
description: "루프 엔지니어링(Ralph Loop) 스킬. 목표가 완수될 때까지 자율적으로 반복 실행하는 루프를 설정, 실행, 제어한다. 사용자가 '루프 실행', 'ralph loop', '끝날 때까지 반복해줘', '자율 루프로 진행해', '루프 엔지니어링', '에러 0건 될 때까지 반복'이라고 요청할 때 사용한다. 완료 약속(completion-promise), 최대 반복 횟수(max-iterations), 회로 차단기(circuit breaker) 및 상태 영속화(state on disk)를 관리한다."
---

# Ralph Loop for Gemini — 자율 루프 엔지니어링

Ralph Loop(Ralph Wiggum Technique)은 **"Iteration > Perfection (반복이 완벽을 이긴다)"**는 철학에 기반하여, 에이전트가 단발성 실패에 굴하지 않고 정의된 목표와 완료 약속(Completion Promise)이 달성될 때까지 디스크에 상태를 기록하며 끈질기게 반복 실행하는 자율 개발 체계입니다.

---

## 1. 핵심 메커니즘

1. **상태의 디스크 영속화 (State on Disk)**:
   - 세션 메모리 대신 `.agents/loop/state.json`과 `.agents/loop/task.md`에 진행 상황을 기록합니다.
2. **명확한 완료 약속 (Completion Promise)**:
   - 사전에 정의된 기계적 토큰(예: `PROMISE_MET: ALL_TESTS_PASSING`, `PROMISE_MET: PRD_COMPLETED`)이 출력될 때만 루프가 안전하게 종료됩니다.
3. **서킷 브레이커 (Circuit Breakers)**:
   - `max_iterations`(기본 10회) 초과 시 강제 종료.
   - 2회 연속 진전이 없을 경우 전략 수정 및 사용자 에스컬레이션.

---

## 2. 루프 실행 방식 (3대 모드)

### 방식 A: Antigravity Stop Hook 세션 인터셉트 (권장)
- **원리**: 에이전트가 턴을 종료하려고 할 때(`model_stop`), Antigravity `hooks.json`의 `Stop` 훅이 이를 가로채 완료 약속이 미충족되었으면 `decision: "continue"`를 반환하여 에이전트를 강제로 깨우고 작업을 이어가게 합니다.
- **설정**:
  1. `.agents/loop/state.json`에 태스크 및 완료 약속 초기화:
     ```json
     {
       "enabled": true,
       "current_iteration": 0,
       "max_iterations": 10,
       "completion_promise": "PROMISE_MET: COMPLETED",
       "task_description": "eCare24 강의 퀴즈 채점 로직 구현 및 전수 테스트 통과"
     }
     ```
  2. `.agents/hooks.json`에 `stop_hook.py` 등록.
  3. 에이전트가 작업을 수행하고, 완료 조건을 만족하면 최종 응답에 약속 토큰(`PROMISE_MET: COMPLETED`)을 포함합니다.

### 방식 B: Fresh Context Bash 셸 러너 (대규모/야간 배치)
- **원리**: 긴 대화로 인한 컨텍스트 오염(Context Rot)을 막기 위해 디스크 상의 상태만 참조하며 독립 셸 루프로 실행.
- **명령어**:
  ```bash
  ./.agents/skills/ralph-loop/scripts/ralph_loop.sh "태스크 설명" 10 "PROMISE_MET: COMPLETED"
  ```

### 방식 C: Antigravity 슬래시 커맨드 `/goal` 결합
- 대화형 인터페이스에서 `/goal`을 입력하여 목표 달성 시까지 에이전트가 자율적으로 도구 실행 및 검증을 멈추지 않도록 지시합니다.

---

## 3. 루프 실행 4단계 워크플로우

### Step 1: 루프 초기화 및 체크포인트
1. Git 작업 상태를 보존합니다 (`git status` 확인).
2. `.agents/loop/task.md`에 체크리스트를 등록합니다:
   ```markdown
   # 태스크 체크리스트
   - [ ] 1. 모델 인터페이스 정의
   - [ ] 2. 퀴즈 계산 함수 구현
   - [ ] 3. 단위 테스트 작성 및 통과
   ```

### Step 2: 루프 가동
- `state.json`을 활성화(`"enabled": true`)하고 목표를 기록합니다.

### Step 3: 원자적 반복 전진 (Atomic Iteration)
- 에이전트는 미완료 태스크 1개를 선택하여 기획/스펙/문서를 작성하고, 즉시 검증을 수행합니다.
- 검증 통과 시 체크리스트를 `[x]`로 갱신하고, **즉시 원자적 Git Commit을 수행**합니다 (`git commit -m "[태그]: [작업내용]"`).
- 실패 시 오류를 진단 데이터로 삼아 즉시 보정합니다.

### Step 4: 완료 및 탈출
- 모든 태스크가 완료되고 검증을 통과하면 최종 정리 커밋(`git commit`)을 수행한 후 `completion_promise` 토큰을 출력합니다.
- `state.json`의 `"enabled"`가 `false`로 전환되며 루프가 공식 종료됩니다.

---

## 4. 루프 비상 정지 (Cancel Loop)

진행 중인 루프를 즉시 중단해야 할 경우:
1. `.agents/loop/state.json`의 `"enabled"` 값을 `false`로 변경합니다.
2. 실행 중인 백그라운드 태스크가 있다면 `manage_task(Action='kill')`로 종료합니다.
