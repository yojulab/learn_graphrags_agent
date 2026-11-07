"""
GraphRAG Agent Web Service & Interactive Teaching Lab
- FastAPI 기반 웹 애플리케이션
- LPG 기반 Neo4j Native Track vs Apache AGE Track 듀얼 백엔드 비교 질의
- Step 1 (기초 질의), Step 2 (벡터 검색), Step 3 (하이브리드 GraphRAG) 지원
- 기존 v1, v2, v3, v4 하위 호환성 유지
- Playwright Headless 데이터 수집 트리거 및 통계 조회
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Demon Slayer Season 1 GraphRAG Interactive Lab",
    description="LPG(Labeled Property Graph) 기반 Neo4j & Apache AGE 듀얼 그래프 데이터베이스 실습 및 하이브리드 검색 웹 서비스",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    version: Optional[str] = "v3"
    track: Optional[str] = "neo4j"   # "neo4j" or "apache_age"
    step: Optional[str] = "step3"    # "step1", "step2", "step3"

class ScrapeRequest(BaseModel):
    url: Optional[str] = "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1"
    headless: bool = True

@app.get("/api/health")
async def health_check():
    """시스템 상태 확인 (Neo4j 및 PostgreSQL Apache AGE 연결 상태)"""
    neo4j_ok = False
    neo4j_error = None
    try:
        from src.neo4j_track.common_neo4j import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            res = session.run("RETURN 1 as ok").single()
            neo4j_ok = res["ok"] == 1
        driver.close()
    except Exception as e:
        neo4j_error = str(e)

    age_ok = False
    age_error = None
    try:
        from src.apache_age_track.common_age import get_age_connection
        conn = get_age_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            age_ok = cur.fetchone()[0] == 1
        conn.close()
    except Exception as e:
        age_error = str(e)

    return {
        "status": "healthy" if (neo4j_ok and age_ok) else "degraded",
        "neo4j_connected": neo4j_ok,
        "neo4j_error": neo4j_error,
        "apache_age_connected": age_ok,
        "apache_age_error": age_error,
        "config": {
            "neo4j_uri": config.NEO4J_URI,
            "neo4j_database": config.NEO4J_DATABASE,
            "postgres_host": config.POSTGRES_HOST,
            "age_graph": config.AGE_GRAPH_NAME,
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        "artifacts": {
            "lpg_knowledge_graph": (BASE_DIR / "output/lpg_knowledge_graph.json").exists(),
            "raw_data_json": (BASE_DIR / "output/raw_data.json").exists(),
            "knowledge_graph_v3": (BASE_DIR / "output/knowledge_graph_v3.json").exists(),
            "knowledge_graph_v4": (BASE_DIR / "output/knowledge_graph_v4.json").exists(),
        }
    }

@app.get("/api/stats")
async def get_stats():
    """지식 그래프 통계 데이터 조회"""
    stats_file = BASE_DIR / "output/statistics_v3.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)

    kg_file = BASE_DIR / "output/lpg_knowledge_graph.json"
    if not kg_file.exists():
        kg_file = BASE_DIR / "output/knowledge_graph_v3.json"
    if kg_file.exists():
        with open(kg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", [])
        rels = data.get("relationships") or data.get("edges", [])
        return {
            "total_nodes": len(nodes),
            "total_relationships": len(rels),
            "node_labels": list(set(n.get("label") for n in nodes if n.get("label"))),
            "relationship_types": list(set(r.get("type") for r in rels if r.get("type")))
        }

    return {"total_nodes": 0, "total_relationships": 0, "node_labels": [], "relationship_types": []}

@app.post("/api/query")
async def execute_query(req: QueryRequest):
    """지식그래프 에이전트 질문 실행"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문 내용을 입력해주세요.")

    try:
        # 트랙/스텝 지정 또는 버전 지정 매핑
        track = (req.track or "").lower()
        step = (req.step or "").lower()
        version = (req.version or "").lower()

        # 버전 지정이 명시된 경우 트랙/스텝으로 변환
        if version == "v1":
            track, step = "neo4j", "step1"
        elif version == "v2":
            track, step = "neo4j", "step2"
        elif version == "v4":
            track, step = "apache_age", "step3"
        elif version == "v3" and not req.track:
            track, step = "neo4j", "step3"

        if track == "apache_age":
            if step == "step1":
                from src.apache_age_track.common_age import get_age_connection, parse_agtype
                conn = get_age_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                        SELECT * FROM cypher('demon_slayer', $$
                            MATCH (a)-[r]-(b)
                            RETURN a.name, type(r), b.name, r.episode_number
                            LIMIT 5
                        $$) as (a agtype, r agtype, b agtype, ep agtype);
                        """)
                        rows = cur.fetchall()
                        answer = "### [Apache AGE Step 1] openCypher 직접 탐색 결과:\n" + "\n".join(
                            f"- [{parse_agtype(r[3])}] {parse_agtype(r[0])} --({parse_agtype(r[1])})--> {parse_agtype(r[2])}"
                            for r in rows
                        )
                finally:
                    conn.close()
            elif step == "step2":
                from src.apache_age_track.step2_indb_vector import search_nodes_by_vector
                nodes = search_nodes_by_vector(question, top_k=3)
                answer = "### [Apache AGE Step 2] In-DB 코사인 유사도 검색 결과:\n" + "\n".join(
                    f"- **{n['name']}** ({n['label']}) | 유사도 점수: {n['similarity']:.4f}"
                    for n in nodes
                )
            else:
                from src.apache_age_track.step3_hybrid_agent import execute_age_hybrid
                answer = execute_age_hybrid(question)
        else:
            # Neo4j Track
            if step == "step1":
                from src.neo4j_track.step1_text2cypher import run_text2cypher
                answer = run_text2cypher(question)
            elif step == "step2":
                from src.neo4j_track.step2_metadata_vector import run_metadata_vector
                answer = run_metadata_vector(question)
            else:
                from src.neo4j_track.step3_hybrid_agent import execute_hybrid_graphrag
                answer = execute_hybrid_graphrag(question)

        return {
            "question": question,
            "version": version or ("v4" if track == "apache_age" else "v3"),
            "track": track,
            "step": step,
            "answer": answer
        }
    except Exception as e:
        traceback.print_exc()
        # 폴백
        return {
            "question": question,
            "version": req.version,
            "track": req.track,
            "step": req.step,
            "answer": f"답변 생성 중 오류가 발생했습니다: {e}"
        }

@app.post("/api/scrape")
async def trigger_scrape(req: ScrapeRequest):
    """Playwright Headless 데이터 수집 트리거"""
    try:
        from src.lpg_builder.scrape_episodes import scrape_episodes_playwright, save_scraped_data, validate_dataset
        episodes = scrape_episodes_playwright(url=req.url, headless=req.headless)
        save_scraped_data(episodes, "output/raw_data.json")
        is_valid = validate_dataset(episodes)
        return {
            "status": "success",
            "episodes_count": len(episodes),
            "validated": is_valid,
            "file": "output/raw_data.json"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>Demon Slayer Season 1 GraphRAG Teaching Lab</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; }
            .header { background: #1e293b; padding: 20px 40px; border-bottom: 1px solid #334155; }
            .container { max-width: 1000px; margin: 40px auto; padding: 0 20px; }
            .card { background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); margin-bottom: 25px; }
            h1 { color: #38bdf8; margin: 0 0 10px 0; font-size: 26px; }
            .subtitle { color: #94a3b8; font-size: 15px; margin-bottom: 20px; }
            .control-panel { display: flex; gap: 15px; margin-bottom: 20px; }
            select, input, button { padding: 12px; border-radius: 6px; border: 1px solid #475569; background: #334155; color: #fff; font-size: 15px; }
            input { flex: 1; }
            button { background: #0284c7; border: none; cursor: pointer; font-weight: bold; transition: background 0.2s; }
            button:hover { background: #0369a1; }
            .result-box { background: #0b1120; padding: 25px; border-radius: 8px; border-left: 4px solid #38bdf8; min-height: 120px; white-space: pre-wrap; line-height: 1.7; font-size: 15px; }
            .badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-bottom: 12px; }
            .badge-neo4j { background: #0284c7; color: #fff; }
            .badge-age { background: #4338ca; color: #fff; }
            .tag { background: #334155; padding: 3px 8px; border-radius: 4px; font-size: 12px; color: #94a3b8; margin-right: 5px; }
            .examples { margin-top: 15px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
            .example-pill { background: #1e293b; border: 1px solid #475569; border-radius: 20px; padding: 6px 14px; font-size: 13px; cursor: pointer; color: #cbd5e1; }
            .example-pill:hover { border-color: #38bdf8; color: #38bdf8; }
        </style>
    </head>
    <body>
        <div class="header">
            <div style="max-width: 1000px; margin: 0 auto;">
                <h1>🗡️ 귀멸의 칼날 시즌 1 GraphRAG Interactive Lab</h1>
                <div class="subtitle">LPG(Labeled Property Graph) 방법론 기반 듀얼 그래프 DB (Neo4j · Apache AGE) 강의 및 비교 실습 대시보드</div>
            </div>
        </div>

        <div class="container">
            <div class="card">
                <div class="control-panel">
                    <div style="display:flex; flex-direction:column; gap:6px;">
                        <span style="font-size:12px; color:#94a3b8; font-weight:bold;">그래프 DB 트랙 선택</span>
                        <select id="trackSelect">
                            <option value="neo4j">🟢 [트랙 A] Neo4j Native Graph</option>
                            <option value="apache_age">🐘 [트랙 B] PostgreSQL & Apache AGE</option>
                        </select>
                    </div>

                    <div style="display:flex; flex-direction:column; gap:6px; flex:1;">
                        <span style="font-size:12px; color:#94a3b8; font-weight:bold;">실습 단계 선택</span>
                        <select id="stepSelect">
                            <option value="step3">Step 3: Hybrid GraphRAG Agent (추천)</option>
                            <option value="step2">Step 2: Metadata / In-DB Vector Search</option>
                            <option value="step1">Step 1: Text2Cypher / openCypher 기초</option>
                        </select>
                    </div>
                </div>

                <div style="display: flex; gap: 10px;">
                    <input type="text" id="questionInput" placeholder="자연어 질문을 입력하세요..." value="카마도 탄지로가 루이와 싸운 에피소드는 몇 화이고 어떤 기술을 썼어?">
                    <button onclick="askQuestion()">질문 전송</button>
                </div>

                <div class="examples">
                    <span style="font-size: 13px; color: #94a3b8;">추천 실습 질문:</span>
                    <span class="example-pill" onclick="setQ(this.innerText)">카마도 탄지로가 루이와 싸운 에피소드는 몇 화이고 어떤 기술을 썼어?</span>
                    <span class="example-pill" onclick="setQ(this.innerText)">토미오카 기유는 시즌 1에서 어떤 활약을 했어?</span>
                    <span class="example-pill" onclick="setQ(this.innerText)">탄지로와 네즈코의 관계를 에피소드별로 정리해줘</span>
                    <span class="example-pill" onclick="setQ(this.innerText)">젠이츠와 이노스케는 언제 처음 만났어?</span>
                </div>
            </div>

            <div class="card">
                <div id="resultCard" class="result-box">
                    <div style="color: #64748b;">상단에서 트랙과 실습 단계를 선택한 후 [질문 전송] 버튼을 클릭하세요.</div>
                </div>
            </div>
        </div>

        <script>
            function setQ(text) {
                document.getElementById('questionInput').value = text;
            }

            async function askQuestion() {
                const q = document.getElementById('questionInput').value;
                const track = document.getElementById('trackSelect').value;
                const step = document.getElementById('stepSelect').value;
                const card = document.getElementById('resultCard');

                card.innerHTML = "⏳ 지식그래프 탐색 및 추론 중... (수 초 소요)";

                try {
                    const res = await fetch('/api/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: q, track: track, step: step})
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || "서버 응답 오류");

                    const badgeClass = track === 'neo4j' ? 'badge-neo4j' : 'badge-age';
                    const trackName = track === 'neo4j' ? 'Neo4j Native Track' : 'Apache AGE Multi-Model Track';
                    card.innerHTML = `<span class="badge ${badgeClass}">${trackName} | ${step.toUpperCase()}</span>\n\n` + data.answer;
                } catch (e) {
                    card.innerHTML = `<span style="color: #ef4444;">❌ 오류: ${e.message}</span>`;
                }
            }
        </script>
    </body>
    </html>
    """
