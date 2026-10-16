# 사용자 소통용 UI 일련번호 태깅 & UI inspect 시스템 규격
## (UI Tagging & Inspect System for GraphRAG Web Explorer)

- **적용 대상**: `src/app/` (FastAPI 웹 대시보드 및 지식그래프 탐색기 UI)
- **적용 목적**: 사용자와 AI 에이전트 간의 웹 UI 피드백 및 Playwright E2E 검증 시, 모호한 묘사 대신 **화면(Page)과 UI 요소(Tag/Element)의 고유 일련번호**를 기반으로 100% 오차 없는 정밀한 커뮤니케이션과 안정적인 셀렉터를 보장함.

---

## 1. 일련번호 부여 규칙 (Serial Numbering Convention)

모든 대화형 UI 요소는 아래 표준 포맷에 따라 유일한 식별 코드를 부여합니다:

```text
[화면코드]-[섹션코드]-[태그/요소코드]
예시: UI-EXPLORER-SEC01-BTN01
```

### 1-1. 화면 및 섹션 코드 체계

| 단위 | 코드 체계 | 설명 및 구성 요소 |
| :--- | :--- | :--- |
| **화면 (Screen)** | `UI-EXPLORER` | GraphRAG 통합 질의응답 및 지식그래프 탐색 화면 |
| **화면 (Screen)** | `UI-METRICS` | Neo4j 및 PostgreSQL AGE 그래프 노드/관계 통계 대시보드 |
| **섹션 (Section)** | `SEC01-QUERY` | 자연어 질문 입력 및 백엔드 선택(Neo4j vs AGE) 섹션 |
| **섹션 (Section)** | `SEC02-RESULT` | 하이브리드 검색 답변 및 생각 태그 정제 결과 표시 섹션 |
| **섹션 (Section)** | `SEC03-GRAPH` | Cytoscape / Vis.js 1~2 hop 서브그래프 시각화 뷰어 섹션 |
| **섹션 (Section)** | `SEC04-CYPHER` | 생성된 Cypher / openCypher 읽기 전용 쿼리 확인 섹션 |
| **요소 (Tag)** | `BTN01` | 질의 실행 버튼 |
| **요소 (Tag)** | `INP01` | 질문 입력창 |
| **요소 (Tag)** | `SEL01` | 백엔드 DB 선택 셀렉트박스 (Neo4j / Apache AGE) |
| **요소 (Tag)** | `CRD01` | 검색된 노드/관계 카드 컴포넌트 |

---

## 2. 단축키 및 UI inspect 뱃지 규격

1. **전역 단축키**: `Alt + I` (Inspect)
2. **원클릭 복사**: 뱃지 클릭 시 4대 정보(일련번호, 페이지명, URI, 항목명)가 클립보드에 즉시 복사됨:
   ```text
   - 일련번호: UI-EXPLORER-SEC01-BTN01
   - 페이지명: GraphRAG 질의 탐색기
   - URI: http://localhost:8000/
   - 항목명: 질문 실행 버튼
   ```
3. **E2E 테스트 연동**: Playwright 자동화 테스트 시 `data-testid` 또는 `data-serial-id` 속성으로 활용하여 깨지지 않는(Flake-free) 셀렉터를 유지함.
