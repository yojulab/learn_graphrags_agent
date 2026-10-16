# Gemini 오케스트레이터 작성 표준 템플릿

본 템플릿은 Antigravity / Gemini 환경에서 다단계 팀 워크플로우를 주도하는 오케스트레이터 지침을 작성할 때 활용한다.

---

## 템플릿 구조

```markdown
# [프로젝트/도메인명] 오케스트레이터

이 문서는 [목표]를 달성하기 위한 에이전트 팀 협업 워크플로우를 오케스트레이션한다.

## 1. 참여 에이전트 및 역할 정의

| 에이전트명 | 모델 티어 | 주 역할 | 산출물 |
| :--- | :--- | :--- | :--- |
| **Architect** | Gemini Pro | 요구사항 분석 및 구조 설계 | `docs/architecture.md` |
| **Builder** | Gemini Flash | 컴포넌트 및 로직 구현 | 소스 코드 파일 |
| **Critic** | Gemini Pro | 적대적 검증 및 결함 판정 | 검증 이슈 목록 |
| **Tester** | Gemini Flash | 테스트 러너 및 브라우저 검증 | 테스트 실행 보고서 |

## 2. 실행 파이프라인 (Phase Workflow)

### Phase 1: 요구사항 및 아키텍처 수립
- **실행 모드**: 반응형 페어 조율 (모드 C)
- **절차**:
  1. 요구사항 명세 확인 및 누락점 파악 (`ask_question` 필요 시 호출)
  2. 시스템 아키텍처 및 데이터 흐름 설계
  3. Artifacts로 설계 다이어그램 및 명세서 발행

### Phase 2: 분할 구현 (Fan-out)
- **실행 모드**: 도구 기반 비동기 오케스트레이션 (모드 A)
- **절차**:
  1. 모듈별 독립 파일 생성 (`write_to_file`)
  2. 단위 로직 구현 및 정적 린트 체크
  3. 완료 시 다음 단계로 산출물 동결(Freeze)

### Phase 3: 품질 및 적대적 교차 검증
- **실행 모드**: 생성-검증 루프 (모드 A + Critic)
- **적용 패턴**: 적대적 검증 (Adversarial Verification) & Loop-until-dry
- **절차**:
  1. Critic이 `confirmed` 결함 목록 도출
  2. Builder가 결함 항목 재수정 (최대 3회 반복)
  3. `run_command`로 빌드 및 테스트 슈트 통과 확인

### Phase 4: UI 인터랙션 검증 (해당 시)
- **실행 모드**: 서브에이전트 위임 (모드 B)
- **절차**:
  1. `browser_subagent`를 기동하여 실 브라우저 동작 검증
  2. 에러 및 UI 깨짐 현상 캡처 및 해결

### Phase 5: 최종 완료 및 회고
- **절차**:
  1. Completeness Critic에 의한 최종 누락 검토
  2. `GEMINI.md` 변경 이력 갱신
  3. 필요 시 `/learn` 또는 `harness-evolve` 스킬 연계
```
