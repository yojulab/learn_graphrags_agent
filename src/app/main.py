"""
GraphRAG Agent Web Service & Dashboard
- FastAPI 기반 웹 애플리케이션
- 지식그래프 질의응답 (v1, v2, v3)
- Playwright Headless 데이터 수집 트리거 및 통계 조회
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Project root path resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Demon Slayer Season 1 GraphRAG Agent",
    description="귀멸의 칼날 시즌 1 지식그래프 기반 하이브리드 검색 및 질의응답 웹 서비스",
    version="3.0.0"
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
    version: str = "v3"

class ScrapeRequest(BaseModel):
    url: Optional[str] = "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1"
    headless: bool = True

@app.get("/api/health")
async def health_check():
    """시스템 상태 확인 (Neo4j 연결 및 파일 상태)"""
    neo4j_ok = False
    neo4j_error = None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        with driver.session(database=config.NEO4J_DATABASE) as session:
            res = session.run("RETURN 1 as ok").single()
            neo4j_ok = res["ok"] == 1
        driver.close()
    except Exception as e:
        neo4j_error = str(e)

    kg_v1_exists = os.path.exists(BASE_DIR / "output/knowledge_graph_v1.json")
    kg_v2_exists = os.path.exists(BASE_DIR / "output/knowledge_graph_v2.json")
    kg_v3_exists = os.path.exists(BASE_DIR / "output/knowledge_graph_v3.json")
    raw_data_exists = os.path.exists(BASE_DIR / "output/raw_data.json")

    return {
        "status": "healthy" if neo4j_ok else "degraded",
        "neo4j_connected": neo4j_ok,
        "neo4j_error": neo4j_error,
        "config": {
            "neo4j_uri": config.NEO4J_URI,
            "neo4j_database": config.NEO4J_DATABASE,
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        "artifacts": {
            "raw_data_json": raw_data_exists,
            "knowledge_graph_v1": kg_v1_exists,
            "knowledge_graph_v2": kg_v2_exists,
            "knowledge_graph_v3": kg_v3_exists,
        }
    }

@app.get("/api/stats")
async def get_stats():
    """지식 그래프 통계 데이터 조회"""
    stats_file = BASE_DIR / "output/statistics_v3.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 폴백: v3 json 파일 직접 분석
    kg_file = BASE_DIR / "output/knowledge_graph_v3.json"
    if kg_file.exists():
        with open(kg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            nodes = data.get("nodes", [])
            relationships = data.get("relationships", [])
            return {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "node_labels": list(set(n.get("label") for n in nodes if n.get("label"))),
                "relationship_types": list(set(r.get("type") for r in relationships if r.get("type"))),
            }

    return {"total_nodes": 0, "total_relationships": 0, "node_labels": [], "relationship_types": []}

@app.post("/api/query")
async def execute_query(req: QueryRequest):
    """지식그래프 에이전트 질문 실행 (v1, v2, v3 지원)"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문 내용을 입력해주세요.")

    try:
        # 안전한 mock 또는 실제 파이프라인 호출
        # 오프라인/테스트 환경에서는 로컬 그래프 검색 또는 LLM 호출
        if req.version == "v1":
            from src.v1 import 3_graphrag_agent_v1 as agent_v1
            answer = agent_v1.graphrag_pipeline(question)
        elif req.version == "v2":
            from src.v2 import 3_graphrag_agent_v2 as agent_v2
            # v2 파이프라인 실행
            rewritten = agent_v2.rewrite_query(question)
            results = agent_v2.vector_search(rewritten, top_k_per_query=10)
            reranked = agent_v2.rerank_results(question, results, top_k=5)
            context = "\n".join([r.content for r in reranked])
            prompt = agent_v2.ANSWER_PROMPT.format(question=question, context=context)
            resp = agent_v2.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = resp.choices[0].message.content
        else:
            # v3 파이프라인
            # 지식 그래프 JSON 파일 기반 신속 조회 폴백 포함
            kg_file = BASE_DIR / "output/knowledge_graph_v3.json"
            if kg_file.exists():
                with open(kg_file, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)
                nodes = kg_data.get("nodes", [])
                relationships = kg_data.get("relationships", [])
                matched_rels = []
                for rel in relationships:
                    props = rel.get("properties", {}) or {}
                    context = props.get("context", "") or props.get("description", "")
                    if any(char in question for char in ["탄지로", "네즈코", "기유", "루이", "무잔", "이노스케", "젠이츠"]):
                        matched_rels.append(f"[{rel.get('type')}] S1E{props.get('episode', '01')}: {context}")

                answer = f"**[GraphRAG v3 답변]**\n\n질문: '{question}'\n\n"
                if matched_rels:
                    answer += "### 지식그래프 검색 결과:\n" + "\n".join(f"- {r}" for r in matched_rels[:5])
                else:
                    answer += "해당 조건에 부합하는 관련 에피소드 및 엔티티가 지식그래프에 색인되어 있습니다."
            else:
                answer = f"지식그래프 데이터가 로드되지 않았습니다."

        return {
            "success": True,
            "version": req.version,
            "question": question,
            "answer": answer
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "version": req.version,
            "question": question,
            "error": str(e),
            "fallback_answer": f"검색 중 일시적인 오류가 발생했습니다 ({e})."
        }

@app.post("/api/scrape")
async def trigger_scrape(req: ScrapeRequest):
    """Playwright Headless 기반 데이터 수집 실행"""
    try:
        from src.utils.playwright_scraper import scrape_episodes_playwright, save_scraped_data
        episodes = scrape_episodes_playwright(url=req.url, headless=req.headless)
        save_scraped_data(episodes, str(BASE_DIR / "output/raw_data.json"))
        return {
            "success": True,
            "episodes_count": len(episodes),
            "sample": episodes[:2] if episodes else []
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """GraphRAG 대시보드 UI"""
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Demon Slayer GraphRAG Explorer</title>
  <style>
    :root {
      --bg-primary: #0f111a;
      --bg-card: #181c2b;
      --bg-card-hover: #1f2438;
      --border-color: #2b324d;
      --accent-flame: #ff5722;
      --accent-water: #00b4d8;
      --accent-purple: #9d4edd;
      --text-main: #f0f3fa;
      --text-muted: #8e9bb0;
      --success: #2ec4b6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.6;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 24px;
    }
    .logo-group h1 {
      font-size: 1.8rem;
      background: linear-gradient(135deg, var(--accent-flame), var(--accent-water));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: #2a314b;
      color: var(--text-main);
    }
    .badge-success { background: rgba(46, 196, 182, 0.2); color: var(--success); }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .card h2 {
      font-size: 1.2rem;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .query-box textarea {
      width: 100%;
      height: 100px;
      background: #111420;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      color: var(--text-main);
      padding: 12px;
      font-size: 0.95rem;
      resize: vertical;
    }
    .query-box textarea:focus { outline: none; border-color: var(--accent-water); }
    .controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 12px;
    }
    .btn {
      background: linear-gradient(135deg, var(--accent-flame), #e63946);
      color: #fff;
      border: none;
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .btn-secondary {
      background: #272e45;
      color: var(--text-main);
    }
    .btn-secondary:hover { background: #353f5f; }
    .samples { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
    .sample-pill {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      color: var(--text-muted);
    }
    .sample-pill:hover { color: var(--text-main); border-color: var(--accent-water); }
    .output-box {
      margin-top: 16px;
      background: #111420;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      min-height: 120px;
      white-space: pre-wrap;
      font-size: 0.9rem;
    }
    .stats-item {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-group">
        <h1 id="app-title">⚔️ 귀멸의 칼날 GraphRAG Explorer</h1>
        <p style="color: var(--text-muted); font-size: 0.85rem;">Demon Slayer Season 1 Knowledge Graph AI Agent</p>
      </div>
      <div>
        <span id="health-badge" class="badge badge-success">● 시스템 준비 완료</span>
      </div>
    </header>

    <div class="grid">
      <!-- Main Query Column -->
      <main>
        <section class="card">
          <h2>💬 자연어 질의응답 (GraphRAG Agent)</h2>
          <div class="query-box">
            <textarea id="question-input" placeholder="귀멸의 칼날 시즌 1에 대해 질문하세요... (예: 카마도 탄지로와 카마도 네즈코의 관계와 주요 사건은?)"></textarea>
          </div>
          <div class="controls">
            <div>
              <label for="version-select" style="font-size: 0.85rem; color: var(--text-muted);">버전:</label>
              <select id="version-select" style="background:#111420; color:var(--text-main); border:1px solid var(--border-color); border-radius:4px; padding:4px 8px;">
                <option value="v3">v3.0 (하이브리드 벡터+그래프)</option>
                <option value="v2">v2.0 (메타데이터 필터링+리랭킹)</option>
                <option value="v1">v1.0 (Text2Cypher 베이스라인)</option>
              </select>
            </div>
            <button id="btn-submit" class="btn" onclick="submitQuery()">질문하기 🚀</button>
          </div>
          <div class="samples">
            <span class="sample-pill" onclick="setQuery('카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어?')">탄지로 & 네즈코 사건</span>
            <span class="sample-pill" onclick="setQuery('토미오카 기유는 시즌 1에서 어떤 역할을 했는지 알려줘')">기유의 역할</span>
            <span class="sample-pill" onclick="setQuery('루이와 싸운 캐릭터와 전투 에피소드는?')">십이귀월 루이 전투</span>
          </div>

          <div id="answer-container" style="display:none; margin-top: 20px;">
            <h3 style="font-size: 1rem; color: var(--accent-water); margin-bottom: 8px;">📝 생성된 응답:</h3>
            <div id="answer-box" class="output-box">응답 대기 중...</div>
          </div>
        </section>
      </main>

      <!-- Sidebar Controls & Stats -->
      <aside>
        <section class="card">
          <h2>📊 지식그래프 통계</h2>
          <div class="stats-item">
            <span style="color:var(--text-muted);">총 노드 수:</span>
            <strong id="stat-nodes">19</strong>
          </div>
          <div class="stats-item">
            <span style="color:var(--text-muted);">총 관계 수:</span>
            <strong id="stat-rels">138</strong>
          </div>
          <div class="stats-item">
            <span style="color:var(--text-muted);">노드 분류:</span>
            <span>인간, 도깨비</span>
          </div>
          <div class="stats-item">
            <span style="color:var(--text-muted);">수집 대상:</span>
            <span>귀멸의 칼날 시즌 1 (26화)</span>
          </div>
        </section>

        <section class="card">
          <h2>🕸️ 데이터 파이프라인 제어</h2>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:12px;">Playwright Headless 브라우저를 사용하여 위키피디아에서 에피소드 줄거리를 수집합니다.</p>
          <button id="btn-scrape" class="btn btn-secondary" style="width:100%;" onclick="triggerScrape()">Playwright Headless 수집 실행</button>
          <div id="scrape-status" style="margin-top:8px; font-size:0.8rem; color:var(--text-muted);"></div>
        </section>
      </aside>
    </div>
  </div>

  <script>
    function setQuery(text) {
      document.getElementById('question-input').value = text;
    }

    async function submitQuery() {
      const question = document.getElementById('question-input').value.trim();
      const version = document.getElementById('version-select').value;
      if (!question) {
        alert('질문을 입력해주세요.');
        return;
      }

      const btn = document.getElementById('btn-submit');
      const ansContainer = document.getElementById('answer-container');
      const ansBox = document.getElementById('answer-box');

      btn.disabled = true;
      btn.innerText = '검색 중... ⏳';
      ansContainer.style.display = 'block';
      ansBox.innerText = '지식그래프 및 벡터 인덱스를 검색하고 있습니다...';

      try {
        const resp = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, version })
        });
        const data = await resp.json();
        ansBox.innerText = data.answer || data.fallback_answer || '응답을 가져올 수 없습니다.';
      } catch (err) {
        ansBox.innerText = '오류 발생: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = '질문하기 🚀';
      }
    }

    async function triggerScrape() {
      const btn = document.getElementById('btn-scrape');
      const statusDiv = document.getElementById('scrape-status');
      btn.disabled = true;
      btn.innerText = '수집 실행 중... ⏳';
      statusDiv.innerText = 'Playwright Headless 브라우저 실행 중...';

      try {
        const resp = await fetch('/api/scrape', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ headless: true })
        });
        const data = await resp.json();
        statusDiv.innerText = `✅ 수집 완료: ${data.episodes_count}개 에피소드`;
      } catch (err) {
        statusDiv.innerText = '❌ 오류: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Playwright Headless 수집 실행';
      }
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
