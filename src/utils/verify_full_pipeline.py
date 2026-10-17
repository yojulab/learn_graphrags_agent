"""
전체 파이프라인 흐름 기반 E2E 검증 및 이슈 해결 스크립트
(Playwright Headless 활용)

[검증 흐름 (Flow)]
1. [Flow 1] Playwright Headless 위키피디아 시즌1 데이터 정밀 수집 및 텍스트/메타데이터 무결성 검증
2. [Flow 2] 듀얼 백엔드(Neo4j 5+ & Apache AGE on PostgreSQL 16+) 접속 및 인덱스/그래프 정합성 검증
3. [Flow 3] GraphRAG 하이브리드 검색 에이전트 (질문 분류 -> 벡터 검색 -> 1~2hop 확장 -> Cypher 템플릿/생성) 검증
4. [Flow 4] FastAPI 웹 대시보드 구동 & Playwright Headless E2E 브라우저 사용자 시나리오 검증 & 스크린샷 캡처
"""

import os
import sys
import json
import time
import re
import asyncio
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

report: Dict[str, Any] = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "overall_status": "PENDING",
    "stages": []
}

def log_stage(title: str):
    print("\n" + "=" * 80, flush=True)
    print(f"🚀 {title}", flush=True)
    print("=" * 80, flush=True)


def stage_1_playwright_data_preparation() -> bool:
    """[Flow 1] Playwright Headless 데이터 수집 및 정밀도 검증"""
    log_stage("[Flow 1] Playwright Headless 데이터 수집 및 정밀도 검증")
    start_time = time.time()
    stage_info = {"stage": "1_prepare_data_playwright", "passed": False, "details": {}}

    try:
        from src.utils.playwright_scraper import validate_dataset

        raw_data_file = BASE_DIR / "output/raw_data.json"
        if not raw_data_file.exists():
            from src.utils.playwright_scraper import scrape_episodes_playwright, save_scraped_data
            episodes = scrape_episodes_playwright()
            save_scraped_data(episodes, str(raw_data_file))
        else:
            with open(raw_data_file, "r", encoding="utf-8") as f:
                episodes = json.load(f)

        assert len(episodes) == 26, f"에피소드 총 개수 불일치: {len(episodes)} != 26"
        assert validate_dataset(episodes) is True, "데이터셋 품질 정밀 검증 실패"

        # knowledge_graph_v3 검증
        kg_file = BASE_DIR / "output/knowledge_graph_v3.json"
        assert kg_file.exists(), f"지식그래프 아티팩트 미존재: {kg_file}"
        with open(kg_file, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        nodes = kg_data.get("nodes", [])
        rels = kg_data.get("relationships", [])
        assert len(nodes) == 19, f"노드 수 불일치: {len(nodes)} != 19"
        assert len(rels) >= 300, f"관계 수 부족: {len(rels)} < 300"

        stage_info["passed"] = True
        stage_info["details"] = {
            "episodes_count": len(episodes),
            "kg_nodes": len(nodes),
            "kg_relationships": len(rels),
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 1 완료] 에피소드 26건 및 노드 19개/관계 {len(rels)}개 정밀 무결성 통과 ({stage_info['details']['duration']})", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 1 실패] {e}", flush=True)

    report["stages"].append(stage_info)
    return stage_info["passed"]


def stage_2_database_integrity() -> bool:
    """[Flow 2] Neo4j 및 PostgreSQL Apache AGE 데이터베이스 정합성 검증"""
    log_stage("[Flow 2] Dual Graph DB (Neo4j & Apache AGE) 정합성 검증")
    start_time = time.time()
    stage_info = {"stage": "2_database_integrity", "passed": False, "details": {}}

    try:
        # 1. Neo4j 검증
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with driver.session(database=config.NEO4J_DATABASE) as session:
            node_cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

            # Vector index count
            idx_records = list(session.run("SHOW VECTOR INDEXES"))
            idx_names = [r["name"] for r in idx_records]

        driver.close()

        assert node_cnt == 19, f"Neo4j 노드 수 불일치: {node_cnt} != 19"
        assert rel_cnt >= 300, f"Neo4j 관계 수 부족: {rel_cnt} < 300"
        assert "entity_embeddings_인간" in idx_names, "entity_embeddings_인간 벡터 인덱스 누락"
        assert "entity_embeddings_도깨비" in idx_names, "entity_embeddings_도깨비 벡터 인덱스 누락"

        # 2. PostgreSQL + Apache AGE 검증
        import psycopg2
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB
        )
        cur = conn.cursor()
        cur.execute("LOAD 'age';")
        cur.execute('SET search_path = ag_catalog, "$user", public;')
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'age';")
        ext = cur.fetchone()
        assert ext is not None and ext[0] == "age", "Apache AGE 확장이 로드되지 않음"
        cur.close()
        conn.close()

        stage_info["passed"] = True
        stage_info["details"] = {
            "neo4j_nodes": node_cnt,
            "neo4j_relationships": rel_cnt,
            "neo4j_vector_indexes": len(idx_names),
            "postgres_age_extension": "LOADED",
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 2 완료] Neo4j (노드 {node_cnt}, 관계 {rel_cnt}, 벡터인덱스 {len(idx_names)}개) 및 Postgres AGE 확장 검증 정상", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 2 실패] {e}", flush=True)

    report["stages"].append(stage_info)
    return stage_info["passed"]


def stage_3_graphrag_agent_execution() -> bool:
    """[Flow 3] GraphRAG 하이브리드 검색 파이프라인 검증 (Neo4j v3 & Apache AGE v4)"""
    log_stage("[Flow 3] Dual GraphRAG 하이브리드 검색 파이프라인 검증 (Neo4j & Apache AGE)")
    start_time = time.time()
    stage_info = {"stage": "3_graphrag_agent_execution", "passed": False, "details": {}}

    try:
        import importlib
        agent_v3 = importlib.import_module("src.neo4j_hybrid.3_graphrag_agent_v3")
        agent_v4_mod = importlib.import_module("src.apache_age.3_graphrag_agent_v4")

        # 1. 쿼리 의도 분류기 검증
        test_question = "카마도 탄지로와 카마도 네즈코는 어떤 관계야?"
        query_type, entities = agent_v3.classify_query_type(test_question)
        print(f"  🎯 질문 의도 분류 결과: {query_type.value}, 엔티티: {entities}", flush=True)
        assert query_type == agent_v3.QueryType.RELATIONSHIP
        assert entities.get("entity_a") == "카마도 탄지로"
        assert entities.get("entity_b") == "카마도 네즈코"

        # 2. Neo4j 하이브리드 리트리버 시드 노드 검색 검증
        seed_nodes = agent_v3.hybrid_retriever.vector_search(test_question)
        assert len(seed_nodes) > 0, "시드 노드 검색 결과가 비어있습니다."

        seed_names = [n["name"] for n in seed_nodes]
        print(f"  🌱 [Neo4j] 벡터 유사도 기반 시드 노드: {seed_names}", flush=True)
        assert any(name in ["카마도 탄지로", "카마도 네즈코"] for name in seed_names)

        # 3. Neo4j 1~2 hop 서브그래프 확장 검증
        seed_ids = [n["id"] for n in seed_nodes]
        subgraph = agent_v3.hybrid_retriever.expand_from_seeds(seed_ids)
        print(f"  🕸️ [Neo4j] 그래프 확장 결과: 서브그래프 레코드 {len(subgraph)}건 확보", flush=True)
        assert len(subgraph) > 0, "서브그래프 확장 결과가 0건입니다."

        # 4. Apache AGE v4 멀티모델 파이프라인 검증
        v4_agent = agent_v4_mod.GraphRAGAgentV4()
        v4_seeds = v4_agent.vector_retriever.search_seed_nodes(test_question, top_k=3)
        assert len(v4_seeds) > 0, "Apache AGE 시드 노드 검색 결과 0건"
        v4_seed_names = [s["name"] for s in v4_seeds]
        print(f"  🐘 [Apache AGE] In-Database 벡터 유사도 시드 노드: {v4_seed_names}", flush=True)
        assert any(name in ["카마도 탄지로", "카마도 네즈코"] for name in v4_seed_names)

        v4_cypher_sql = v4_agent.cypher_engine.generate_intent_query(
            agent_v4_mod.QueryType.RELATIONSHIP,
            {"entity_a": "카마도 탄지로", "entity_b": "카마도 네즈코"},
            v4_seeds
        )
        v4_records = v4_agent.cypher_engine.execute_cypher_sql(v4_cypher_sql)
        print(f"  🕸️ [Apache AGE] openCypher in SQL 매칭 레코드: {len(v4_records)}건", flush=True)
        assert len(v4_records) > 0, "Apache AGE openCypher 질의 결과 0건"

        stage_info["passed"] = True
        stage_info["details"] = {
            "test_question": test_question,
            "query_type": query_type.value,
            "entities": entities,
            "neo4j_seed_nodes": seed_names,
            "neo4j_subgraph_records": len(subgraph),
            "age_seed_nodes": v4_seed_names,
            "age_cypher_records": len(v4_records),
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 3 완료] Dual GraphRAG (Neo4j & Apache AGE) 검색 및 확장 파이프라인 정상 ({stage_info['details']['duration']})", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 3 실패] {e}", flush=True)

    report["stages"].append(stage_info)
    return stage_info["passed"]


def stage_4_playwright_e2e_web_ui() -> bool:
    """[Flow 4] FastAPI 웹 대시보드 및 Playwright Headless 실 브라우저 E2E 검증"""
    log_stage("[Flow 4] FastAPI 대시보드 & Playwright Headless 실 브라우저 E2E 검증")
    start_time = time.time()
    stage_info = {"stage": "4_web_ui_playwright_e2e", "passed": False, "details": {}}

    try:
        from src.web_app.main import serve_ui, health_check, get_stats, execute_query, QueryRequest

        # 1. FastAPI UI 렌더링 검증
        html_response = asyncio.run(serve_ui())
        html_content = html_response.body.decode("utf-8") if isinstance(html_response.body, bytes) else str(html_response.body)

        screenshot_path = BASE_DIR / "output/playwright_e2e_screenshot.png"

        # 2. Playwright Headless 브라우저 실행 및 DOM 컴포넌트 검증
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.set_content(html_content)

            # UI 타이틀 검증
            title_text = page.locator("#app-title").inner_text()
            assert "귀멸의 칼날 GraphRAG" in title_text
            print(f"  📄 대시보드 타이틀 확인: {title_text}", flush=True)

            # 상태 배지 검증
            health_badge = page.locator("#health-badge").inner_text()
            assert "시스템 준비 완료" in health_badge
            print(f"  🩺 시스템 상태 배지: {health_badge}", flush=True)

            # 통계 엘리먼트 검증
            stat_nodes = int(page.locator("#stat-nodes").inner_text())
            stat_rels = int(page.locator("#stat-rels").inner_text())
            assert stat_nodes == 19, f"UI 노드 수 불일치: {stat_nodes} != 19"
            assert stat_rels >= 300, f"UI 관계 수 부족: {stat_rels} < 300"
            print(f"  📊 UI 통계 카운터: 노드 {stat_nodes}개, 관계 {stat_rels}개", flush=True)

            # 버전 선택 옵션 검증 (v4, v3 포함 여부)
            version_options = page.locator("#version-select option").all_inner_texts()
            assert any("v4.0" in opt for opt in version_options), f"v4.0 옵션 누락: {version_options}"
            print(f"  🎛️  버전 선택 옵션: {version_options}", flush=True)

            # 입력 폼 인터랙션 시뮬레이션
            question_input = page.locator("#question-input")
            question_input.fill("카마도 탄지로와 카마도 네즈코의 여정")

            version_select = page.locator("#version-select")
            version_select.select_option("v4")

            btn_submit = page.locator("#btn-submit")
            assert btn_submit.is_visible()

            # Playwright Headless 전체 화면 캡처 저장
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  📸 E2E 브라우저 스크린샷 캡처 완료: {screenshot_path}", flush=True)

            browser.close()

        # 3. 비동기 엔드포인트 핸들러 E2E 검증 (v3 및 v4 모두 검증)
        health_res = asyncio.run(health_check())
        assert health_res["status"] == "healthy", f"헬스체크 실패: {health_res}"
        assert health_res["neo4j_connected"] is True, "Neo4j 연결 미확인"

        stats_res = asyncio.run(get_stats())
        assert stats_res["total_nodes"] == 19

        # v3 쿼리
        query_res_v3 = asyncio.run(execute_query(QueryRequest(question="탄지로와 네즈코", version="v3")))
        assert query_res_v3["success"] is True
        print(f"  💬 [v3] 에이전트 응답 검증 완료: {query_res_v3['answer'][:60]}...", flush=True)

        # v4 쿼리
        query_res_v4 = asyncio.run(execute_query(QueryRequest(question="탄지로와 네즈코", version="v4")))
        assert query_res_v4["success"] is True
        print(f"  💬 [v4] 에이전트 응답 검증 완료: {query_res_v4['answer'][:60]}...", flush=True)

        stage_info["passed"] = True
        stage_info["details"] = {
            "ui_nodes": stat_nodes,
            "ui_relationships": stat_rels,
            "api_health": health_res["status"],
            "query_result_sample": query_res_v4["answer"][:100],
            "screenshot_path": str(screenshot_path),
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 4 완료] Playwright Headless Web UI E2E 인터랙션 및 스크린샷 검증 통과 ({stage_info['details']['duration']})", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 4 실패] {e}", flush=True)

    report["stages"].append(stage_info)
    return stage_info["passed"]


def main():
    print("\n" + "#" * 80, flush=True)
    print("🎬 GraphRAG 전체 파이프라인 흐름 기반 E2E 검증 (Playwright Headless)", flush=True)
    print("#" * 80, flush=True)

    s1 = stage_1_playwright_data_preparation()
    s2 = stage_2_database_integrity()
    s3 = stage_3_graphrag_agent_execution()
    s4 = stage_4_playwright_e2e_web_ui()

    all_passed = s1 and s2 and s3 and s4
    report["overall_status"] = "PASS" if all_passed else "FAIL"

    report_path = BASE_DIR / "output/pipeline_flow_e2e_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("📊 전체 파이프라인 종합 검증 보고", flush=True)
    print("=" * 80, flush=True)
    for s in report["stages"]:
        status_mark = "✅ PASS" if s["passed"] else "❌ FAIL"
        print(f"  - {s['stage']:<32}: {status_mark}", flush=True)
    print(f"\n최종 결과: {'🎉 ALL STAGES PASSED' if all_passed else '⚠️ SOME STAGES FAILED'}", flush=True)
    print(f"결과 보고서: {report_path}", flush=True)
    print("=" * 80 + "\n", flush=True)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
