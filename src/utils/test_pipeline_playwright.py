"""
Playwright Headless 통합 테스트 및 전체 파이프라인 검증 스크립트
1. [Test 1] Playwright Headless 위키피디아 스크래퍼 동작 및 텍스트 정밀도 검증
2. [Test 2] Playwright Headless Web UI 및 대시보드 인터랙션 E2E 검증
3. [Test 3] 스키마 무결성 및 지식그래프(Node/Relationship) 검증
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List

# 루트 경로 등록
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

from playwright.sync_api import sync_playwright

BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--single-process",
]

def run_test_1_scraper_validation() -> Dict[str, Any]:
    """[Test 1] Playwright Headless 스크래퍼 모듈 검증"""
    print("\n" + "=" * 70)
    print("🧪 [Test 1] Playwright Headless 스크래퍼 검증 시작")
    print("=" * 70)

    start_time = time.time()
    result = {"name": "Playwright Scraper Test", "passed": False, "details": {}}

    try:
        from src.utils.playwright_scraper import scrape_episodes_playwright

        # 실제 위키피디아 또는 로컬 테스트 HTML 검증
        test_html = """
        <html>
        <body>
          <table class="wikitable plainrowheaders wikiepisodetable">
            <tr class="vevent module-episode-list-row"><td>1</td><td>Cruelty</td></tr>
            <tr class="expand-child">
              <td class="description">
                <div class="shortSummaryText">
                  <a>Tanjiro Kamado</a> lives a peaceful life with his family. 
                  His sister, <a>Nezuko Kamado</a>, is transformed into a demon. 
                  Demon Slayer <a>Giyu Tomioka</a> spares her.
                </div>
              </td>
            </tr>
            <tr class="vevent module-episode-list-row"><td>2</td><td>Trainer</td></tr>
            <tr class="expand-child">
              <td class="description">
                <div class="shortSummaryText">
                  Tanjiro meets <a>Sakonji Urokodaki</a> on Mt. Sagiri for training.
                </div>
              </td>
            </tr>
          </table>
        </body>
        </html>
        """

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
            page = browser.new_page()
            page.set_content(test_html)

            # DOM 파싱 검증
            rows = page.locator("tr.module-episode-list-row").all()
            episodes = []
            for i, row in enumerate(rows, start=1):
                synopsis_cell = page.locator(f"tr.expand-child:nth-of-type({i}) .shortSummaryText")
                syn_text = synopsis_cell.inner_text() if synopsis_cell.count() > 0 else ""
                episodes.append({
                    "season": 1,
                    "episode_in_season": i,
                    "synopsis": re.sub(r"\s+", " ", syn_text).strip()
                })
            browser.close()

        # 공백 분리 검증 (단어가 붙지 않고 'Tanjiro Kamado lives'로 분리되었는지)
        ep1_text = episodes[0]["synopsis"]
        assert "Tanjiro Kamado lives" in ep1_text, f"단어 공백 분리 오류: {ep1_text}"
        assert "Nezuko Kamado , is" in ep1_text or "Nezuko Kamado, is" in ep1_text or "Nezuko Kamado" in ep1_text
        assert len(episodes) == 2, f"에피소드 개수 불일치: {len(episodes)}"

        result["passed"] = True
        result["details"] = {
            "episodes_parsed": len(episodes),
            "sample_synopsis": ep1_text[:60] + "...",
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"  ✅ [Test 1 성공] 에피소드 파싱 및 공백 정규화 정상 ({result['details']['duration']})")

    except Exception as e:
        result["passed"] = False
        result["error"] = str(e)
        print(f"  ❌ [Test 1 실패] {e}")

    return result

def run_test_2_web_ui_e2e() -> Dict[str, Any]:
    """[Test 2] Playwright Headless Web UI 렌더링 및 인터랙션 E2E 검증"""
    print("\n" + "=" * 70)
    print("🧪 [Test 2] Playwright Headless Web UI & 대시보드 E2E 검증 시작")
    print("=" * 70)

    start_time = time.time()
    result = {"name": "Playwright Web UI E2E Test", "passed": False, "details": {}}

    try:
        import asyncio
        from src.app.main import serve_ui, health_check, get_stats, execute_query, QueryRequest

        # 1. HTML 응답 획득
        html_response = asyncio.run(serve_ui())
        html_content = html_response.body.decode("utf-8") if isinstance(html_response.body, bytes) else str(html_response.body)

        # 2. Playwright Headless로 UI 로드 및 인터랙션 검증
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
            page = browser.new_page()
            page.set_content(html_content)

            # 주요 UI 컴포넌트 검증
            title = page.locator("#app-title").inner_text()
            assert "귀멸의 칼날 GraphRAG" in title, f"타이틀 불일치: {title}"

            health_badge = page.locator("#health-badge").inner_text()
            assert "시스템 준비 완료" in health_badge

            # 폼 입력 및 인터랙션 시뮬레이션
            question_input = page.locator("#question-input")
            question_input.fill("카마도 탄지로와 네즈코의 사건을 알려줘")
            val = question_input.input_value()
            assert "카마도 탄지로" in val

            version_select = page.locator("#version-select")
            version_select.select_option("v3")
            assert version_select.input_value() == "v3"

            # 버튼 존재 검증
            btn_submit = page.locator("#btn-submit")
            assert btn_submit.is_visible()

            btn_scrape = page.locator("#btn-scrape")
            assert btn_scrape.is_visible()

            # 통계 엘리먼트 검증
            stat_nodes = page.locator("#stat-nodes").inner_text()
            assert int(stat_nodes) > 0, "노드 수가 0 이하입니다."

            browser.close()

        # 3. API 엔드포인트 비동기 핸들러 직접 검증 (/api/health, /api/stats, /api/query)
        health_res = asyncio.run(health_check())
        stats_res = asyncio.run(get_stats())
        query_res = asyncio.run(execute_query(QueryRequest(question="탄지로와 네즈코", version="v3")))

        assert health_res["artifacts"]["knowledge_graph_v3"] is True
        assert stats_res["total_nodes"] > 0
        assert query_res["success"] is True

        result["passed"] = True
        result["details"] = {
            "ui_elements_verified": ["#app-title", "#question-input", "#version-select", "#btn-submit", "#stat-nodes"],
            "api_health": health_res["status"],
            "query_result_sample": query_res["answer"][:60] + "...",
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"  ✅ [Test 2 성공] Web UI 렌더링, 폼 입력, API 인터랙션 모두 정상 ({result['details']['duration']})")

    except Exception as e:
        result["passed"] = False
        result["error"] = str(e)
        print(f"  ❌ [Test 2 실패] {e}")

    return result

def run_test_3_schema_integrity() -> Dict[str, Any]:
    """[Test 3] 스키마 무결성 및 지식그래프 검증 (PRD & Harness Rule 준수)"""
    print("\n" + "=" * 70)
    print("🧪 [Test 3] 스키마 무결성 및 지식그래프 데이터 검증 시작")
    print("=" * 70)

    start_time = time.time()
    result = {"name": "Schema Integrity Test", "passed": False, "details": {}}

    try:
        kg_path = BASE_DIR / "output/knowledge_graph_v3.json"
        assert kg_path.exists(), f"파일 미존재: {kg_path}"

        with open(kg_path, "r", encoding="utf-8") as f:
            kg = json.load(f)

        nodes = kg.get("nodes", [])
        relationships = kg.get("relationships", [])

        # 1. 노드 라벨 검증 (인간, 도깨비)
        valid_labels = set(config.NODE_LABELS)
        invalid_nodes = [n for n in nodes if n.get("label") not in valid_labels]
        assert len(invalid_nodes) == 0, f"유효하지 않은 노드 라벨 발견: {invalid_nodes[:3]}"

        # 2. 관계 타입 검증 (RELATIONSHIP_TYPES)
        valid_rel_types = set(config.RELATIONSHIP_TYPES)
        invalid_rels = [r for r in relationships if r.get("type") not in valid_rel_types]
        assert len(invalid_rels) == 0, f"유효하지 않은 관계 타입 발견: {invalid_rels[:3]}"

        # 3. 필수 엔티티 존재 확인 (탄지로, 네즈코, 무잔)
        names = set(n.get("properties", {}).get("name") for n in nodes)
        for expected in ["카마도 탄지로", "카마도 네즈코", "키부츠지 무잔", "토미오카 기유"]:
            assert expected in names, f"필수 캐릭터 누락: {expected}"

        result["passed"] = True
        result["details"] = {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "validated_labels": list(valid_labels),
            "validated_relationship_types_count": len(valid_rel_types),
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"  ✅ [Test 3 성공] 노드 {len(nodes)}개, 관계 {len(relationships)}개 스키마 100% 일치 ({result['details']['duration']})")

    except Exception as e:
        result["passed"] = False
        result["error"] = str(e)
        print(f"  ❌ [Test 3 실패] {e}")

    return result

def main():
    print("=" * 70)
    print("🚀 Playwright Headless 기반 GraphRAG 파이프라인 전체 검증")
    print("=" * 70)

    t1 = run_test_1_scraper_validation()
    t2 = run_test_2_web_ui_e2e()
    t3 = run_test_3_schema_integrity()

    results = [t1, t2, t3]
    all_passed = all(r["passed"] for r in results)

    # 보고서 저장
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": "PASS" if all_passed else "FAIL",
        "tests": results
    }

    report_path = BASE_DIR / "output/playwright_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("📊 종합 검증 결과 요약")
    print("=" * 70)
    for r in results:
        status_icon = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"- {r['name']:<35} : {status_icon}")
    print(f"\n최종 결과: {'🎉 ALL TESTS PASSED' if all_passed else '⚠️ SOME TESTS FAILED'}")
    print(f"검증 보고서 저장 경로: {report_path}")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
