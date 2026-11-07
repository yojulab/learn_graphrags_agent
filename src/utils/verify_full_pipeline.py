"""
전체 파이프라인 흐름 기반 E2E 검증 및 무결성 진단 스크립트
(Playwright Headless & Dual Graph DB LPG 파이프라인)

[검증 흐름 (Flow)]
1. [Flow 1] Playwright Headless 위키피디아 시즌1 데이터 정밀 수집 및 LPG 무결성 검증
2. [Flow 2] 듀얼 백엔드(Neo4j 5+ & Apache AGE on PostgreSQL 16+) 접속 및 인덱스/그래프 정합성 검증
3. [Flow 3] GraphRAG 하이브리드 검색 에이전트 (Neo4j Track & Apache AGE Track) 검증
4. [Flow 4] FastAPI 웹 대시보드 구동 & Playwright Headless E2E 브라우저 사용자 시나리오 검증
"""

import os
import sys
import json
import time
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List

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
    """[Flow 1] Playwright Headless 데이터 수집 및 LPG 정밀도 검증"""
    log_stage("[Flow 1] Playwright Headless 데이터 수집 및 LPG 정밀도 검증")
    start_time = time.time()
    stage_info = {"stage": "1_prepare_data_playwright", "passed": False, "details": {}}

    try:
        from src.lpg_builder.scrape_episodes import validate_dataset

        raw_data_file = BASE_DIR / "output/raw_data.json"
        if not raw_data_file.exists():
            from src.lpg_builder.scrape_episodes import scrape_episodes_playwright, save_scraped_data
            episodes = scrape_episodes_playwright()
            save_scraped_data(episodes, str(raw_data_file))
        else:
            with open(raw_data_file, "r", encoding="utf-8") as f:
                episodes = json.load(f)

        assert len(episodes) == 26, f"에피소드 총 개수 불일치: {len(episodes)} != 26"
        assert validate_dataset(episodes) is True, "데이터셋 품질 정밀 검증 실패"

        # lpg_knowledge_graph.json 또는 knowledge_graph_v3 검증
        kg_file = BASE_DIR / "output/lpg_knowledge_graph.json"
        if not kg_file.exists():
            from src.lpg_builder.extract_lpg_production import build_production_lpg
            build_production_lpg()

        with open(kg_file, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        nodes = kg_data.get("nodes", [])
        rels = kg_data.get("relationships") or kg_data.get("edges", [])
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
        from src.neo4j_track.common_neo4j import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            node_cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            idx_records = list(session.run("SHOW VECTOR INDEXES"))
            idx_names = [r["name"] for r in idx_records]
        driver.close()

        assert node_cnt == 19, f"Neo4j 노드 수 불일치: {node_cnt} != 19"
        assert rel_cnt >= 300, f"Neo4j 관계 수 부족: {rel_cnt} < 300"
        assert any("인간" in n for n in idx_names), "인간 노드 벡터 인덱스 누락"

        # 2. PostgreSQL + Apache AGE 검증
        from src.apache_age_track.common_age import get_age_connection
        conn = get_age_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'age';")
            ext = cur.fetchone()
            assert ext is not None and ext[0] == "age", "Apache AGE 확장이 로드되지 않음"
            cur.execute("SELECT count(*) FROM demon_slayer_node_embeddings;")
            age_node_cnt = cur.fetchone()[0]
            assert age_node_cnt == 19, f"AGE 노드 임베딩 수 불일치: {age_node_cnt} != 19"
        conn.close()

        stage_info["passed"] = True
        stage_info["details"] = {
            "neo4j_nodes": node_cnt,
            "neo4j_relationships": rel_cnt,
            "neo4j_vector_indexes": len(idx_names),
            "age_nodes": age_node_cnt,
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 2 완료] Neo4j (노드 {node_cnt}, 관계 {rel_cnt}) 및 Postgres AGE (노드 {age_node_cnt}) 정상", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 2 실패] {e}", flush=True)

    report["stages"].append(stage_info)
    return stage_info["passed"]

def stage_3_graphrag_agent_execution() -> bool:
    """[Flow 3] GraphRAG 하이브리드 검색 파이프라인 검증 (Neo4j Track & Apache AGE Track)"""
    log_stage("[Flow 3] Dual GraphRAG 하이브리드 검색 파이프라인 검증 (Neo4j & Apache AGE)")
    start_time = time.time()
    stage_info = {"stage": "3_graphrag_agent_execution", "passed": False, "details": {}}

    try:
        # 1. Neo4j Track 의도 분류 & 시드 검색 검증
        from src.neo4j_track.step3_hybrid_agent import classify_query_type, QueryType, vector_seed_search, expand_subgraph_from_seeds
        test_question = "카마도 탄지로와 카마도 네즈코는 어떤 관계야?"
        query_type, entities = classify_query_type(test_question)
        print(f"  🎯 [Neo4j] 질문 의도 분류 결과: {query_type.value}, 개체: {entities}", flush=True)
        assert query_type == QueryType.RELATIONSHIP
        assert entities.get("entity_a") == "카마도 탄지로"

        seed_nodes = vector_seed_search(test_question, top_k=3)
        assert len(seed_nodes) > 0, "Neo4j 시드 노드 검색 결과 0건"
        seed_names = [n["name"] for n in seed_nodes]
        print(f"  🌱 [Neo4j] 시드 노드: {seed_names}", flush=True)

        seed_ids = [n["id"] for n in seed_nodes]
        subgraph = expand_subgraph_from_seeds(seed_ids, max_depth=2)
        print(f"  🕸️ [Neo4j] 1~2 Hop 서브그래프 확장: {len(subgraph)}건", flush=True)
        assert len(subgraph) > 0, "Neo4j 서브그래프 확장 0건"

        # 2. Apache AGE Track 인-데이터베이스 벡터 & 확장 검증
        from src.apache_age_track.step3_hybrid_agent import search_seed_nodes, expand_age_subgraph
        age_seeds = search_seed_nodes(test_question, top_k=3)
        assert len(age_seeds) > 0, "Apache AGE 시드 노드 검색 0건"
        age_seed_names = [s["name"] for s in age_seeds]
        print(f"  🐘 [Apache AGE] In-DB 시드 노드: {age_seed_names}", flush=True)

        age_subgraph = expand_age_subgraph(age_seed_names)
        print(f"  🕸️ [Apache AGE] openCypher 1~2 Hop 확장: {len(age_subgraph)}건", flush=True)
        assert len(age_subgraph) > 0, "Apache AGE 확장 0건"

        stage_info["passed"] = True
        stage_info["details"] = {
            "test_question": test_question,
            "neo4j_seed_nodes": seed_names,
            "neo4j_subgraph_count": len(subgraph),
            "age_seed_nodes": age_seed_names,
            "age_subgraph_count": len(age_subgraph),
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 3 완료] Dual GraphRAG 트랙 (Neo4j & Apache AGE) 검색/확장 검증 정상 ({stage_info['details']['duration']})", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 3 실패] {e}", flush=True)

    report["stages"].append(stage_info)
    return stage_info["passed"]

def stage_4_playwright_e2e_web_ui() -> bool:
    """[Flow 4] FastAPI 웹 대시보드 및 Playwright Headless 실 브라우저 E2E 검증"""
    log_stage("[Flow 4] FastAPI Web UI & Playwright 브라우저 E2E 사용자 시나리오 검증")
    start_time = time.time()
    stage_info = {"stage": "4_playwright_e2e_web_ui", "passed": False, "details": {}}

    import subprocess
    server_process = None
    try:
        # FastAPI 서버 백그라운드 구동 확인
        port = 8000
        import urllib.request
        server_ready = False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    server_ready = True
        except Exception:
            pass

        if not server_ready:
            print(f"  ⏳ Uvicorn 웹 서버 가동 중 (포트 {port})...", flush=True)
            server_process = subprocess.Popen(
                ["uvicorn", "src.web_app.main:app", "--host", "0.0.0.0", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            for _ in range(15):
                time.sleep(1)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as resp:
                        if resp.status == 200:
                            server_ready = True
                            break
                except Exception:
                    pass

        assert server_ready, "FastAPI 웹 서버가 준비되지 않았습니다."
        print(f"  🌐 웹 서버 가동 확인 완료: http://127.0.0.1:{port}", flush=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
            page = browser.new_page(viewport={"width": 1280, "height": 800})

            # 1. 대시보드 접속
            page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
            page_title = page.title()
            print(f"  📄 페이지 로드 확인: '{page_title}'", flush=True)
            assert "GraphRAG" in page_title, "페이지 타이틀 오류"

            # 2. 질문 입력 및 질의 전송
            q_input = page.locator("#questionInput")
            q_input.fill("카마도 탄지로가 루이와 싸운 에피소드는 몇 화이고 어떤 기술을 썼어?")
            send_btn = page.locator("button:has-text('질문 전송')")
            send_btn.click()

            # 3. 답변 렌더링 대기
            result_card = page.locator("#resultCard")
            result_card.wait_for(timeout=30000)
            time.sleep(3)
            result_text = result_card.inner_text()

            # 4. 스크린샷 캡처
            screenshot_path = BASE_DIR / "output/playwright_e2e_screenshot.png"
            page.screenshot(path=str(screenshot_path))
            print(f"  📸 E2E 브라우저 스크린샷 저장 완료: {screenshot_path}", flush=True)

            browser.close()

        stage_info["passed"] = True
        stage_info["details"] = {
            "web_title": page_title,
            "result_preview": result_text[:120].replace("\n", " "),
            "screenshot": str(screenshot_path),
            "duration": f"{time.time() - start_time:.2f}s"
        }
        print(f"✅ [Flow 4 완료] Playwright 브라우저 E2E 검증 100% 정상 ({stage_info['details']['duration']})", flush=True)

    except Exception as e:
        stage_info["error"] = str(e)
        print(f"❌ [Flow 4 실패] {e}", flush=True)
    finally:
        if server_process:
            server_process.terminate()

    report["stages"].append(stage_info)
    return stage_info["passed"]

def run_full_pipeline_verification():
    print("=" * 80)
    print("🚀 [E2E Pipeline] LPG & Dual GraphRAG 전체 파이프라인 검증 가동")
    print("=" * 80)

    f1 = stage_1_playwright_data_preparation()
    f2 = stage_2_database_integrity()
    f3 = stage_3_graphrag_agent_execution()
    f4 = stage_4_playwright_e2e_web_ui()

    all_passed = f1 and f2 and f3 and f4
    report["overall_status"] = "PASSED" if all_passed else "FAILED"

    report_path = BASE_DIR / "output/pipeline_flow_e2e_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"📊 최종 E2E 파이프라인 검증 결과: {report['overall_status']}")
    print(f"📄 종합 리포트 저장: {report_path}")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = run_full_pipeline_verification()
    sys.exit(0 if success else 1)
