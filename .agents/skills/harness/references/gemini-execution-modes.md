# Gemini & Antigravity 실행 모드 상세 가이드

본 문서는 Claude Code의 멀티에이전트 프리미티브(`Workflow`, `Agent`, `SendMessage`)를 **Google Antigravity 및 Gemini 실행 환경**의 네이티브 도구 체계로 완전히 재구성한 3대 실행 모드를 정의한다.

---

## 1. 개요: 환경별 실행 프리미티브 비교

| 기능 축 | Claude Code 환경 (v2) | Google Antigravity & Gemini 환경 |
| :--- | :--- | :--- |
| **결정론적 다단계 실행** | `Workflow` 스크립트 (`pipeline`, `parallel`) | **도구 기반 비동기 파이프라인 & 백그라운드 태스크** (`run_command` Persistent/Daemon, `manage_task`, `schedule`) |
| **전문가 상호작용** | `Agent(name:)` + `SendMessage` | **서브에이전트 위임 & 반응형 페어 프로그래밍** (`browser_subagent`, `subagent delegation`, Artifacts 상호작용) |
| **이벤트 알림 / 대기** | `TaskWait`, `Workflow` 리스너 | **반응형 웨이크업 (Reactive Wakeup)**: 태스크 완료/메시지 수신 시 무한 루프 폴링 없이 자동 재개 |
| **상태 보존 / 공유** | Task API (`TaskCreate`, `TaskUpdate`) | **지속형 터미널 세션 (`RunPersistent`) & 아티팩트 시스템 (`Artifacts`)** |
| **목표 지향 자율 실행** | `/loop`, `/workflow` | **`/goal` (장기 자율 달성), `/plan` (단계별 계획), `/grill-me` (사전 조율)** |

---

## 2. 3대 Gemini 실행 모드

### 모드 A: 도구 기반 비동기 오케스트레이션 (Tool-based Async Orchestration)
- **개념**: 빌드, 린트, 테스트, 배치 스크립트 실행, 대규모 파일 변환 등 **입출력이 명확하고 결정론적인 절차**를 Antigravity의 비동기 실행 도구로 지휘한다.
- **핵심 프리미티브**:
  - `run_command(IsDaemon=false, WaitMsBeforeAsync=5000)`: 백그라운드 프로세스 실행 및 결과 수신
  - `run_command(RunPersistent=true)`: 세션 변수와 상태를 보존하는 지속형 터미널 운영
  - `manage_task`: 백그라운드 작업 상태 확인, 표준 입력 전송, 취소 관리
  - `schedule(TimerCondition='any' | 'task-id')`: 지연 실행 및 하트비트 모니터링
- **적합한 작업**: 대규모 리팩터링, 전수 테스트 러너, 빌드 파이프라인, 정적 분석 스윕.

### 모드 B: 서브에이전트 및 브라우저 위임 (Subagent Delegation)
- **개념**: 사용자 인터페이스 검증, 웹 리서치, 격리된 특수 태스크를 전문 서브에이전트에 위임하고 완결된 결과 보고서를 수취한다.
- **핵심 프리미티브**:
  - `browser_subagent`: 헤드리스 브라우저를 통한 E2E 테스트, DOM 검사, 웹 UI 인터랙션 검증 및 WebP 비디오 녹화
  - 태스크 스코프별 격리된 하위 작업 수행
- **적합한 작업**: UI 렌더링 검증, 사용자 여정(User Flow) E2E 테스트, 외부 API/문서 수집.

### 모드 C: 반응형 페어 프로그래밍 & 대화형 조율 (Reactive Pair Orchestration)
- **개념**: 복잡한 설계 결정, 비즈니스 로직 작성, 다자간 인터랙션이 필요한 상황에서 사용자와 깊이 있게 조율하며 실시간 피드백을 반영한다.
- **핵심 프리미티브**:
  - **반응형 웨이크업 (Reactive Wakeup)**: 백그라운드 작업 완료 시 자동으로 깨어나 다음 단계 실행
  - **아티팩트 시스템 (Artifacts)**: 설계서, 아키텍처 다이어그램(Mermaid), 진행 보고서를 구조화하여 공유
  - **인터랙티브 질문 (`ask_question`)**: 모호한 요구사항에 대한 다중 선택지 모달 제시
  - **슬래시 커맨드 결합**:
    - `/goal`: 사용자의 지속 개입 없이 완결까지 스스로 루프를 돌며 문제 해결
    - `/plan`: 아키텍처와 단계별 실행 계획 사전 합의
    - `/grill-me`: 에이전트 주도의 질문 공세를 통해 설계 의사결정 명확화
    - `/learn`: 성공 경험과 룰을 프로젝트 지식베이스에 영속화

---

## 3. 실행 모드 선택 결정 트리

```
작업의 성격이 무엇인가?
│
├── 제어 흐름이 명확하고 명령어나 스크립트로 검증 가능한가?
│   ├── YES ──▶ [모드 A: 도구 기반 비동기 오케스트레이션]
│   │           (run_command, background task, schedule)
│   │
│   └── NO
│       ├── UI 검증, 실제 브라우저 조작 또는 독립된 리서치가 필요한가?
│       │   ├── YES ──▶ [모드 B: 서브에이전트 및 브라우저 위임]
│       │   │           (browser_subagent, isolated task)
│       │   │
│       │   └── NO ───▶ [모드 C: 반응형 페어 프로그래밍 & 대화형 조율]
│       │               (ask_question, Artifacts, /plan, /goal)
```
