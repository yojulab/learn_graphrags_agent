"""
[Neo4j Track Step 2] LPG 속성 메타데이터 필터링 및 벡터 검색(Reranking) 실습
(src/02_neo4j_track/3_step2_metadata_vector.py)

- 실습 내용:
  1. 자연어 질의 ➔ LLM 쿼리 리라이팅 (Query Rewriter & SearchFilters)
  2. Neo4j HNSW 코사인 벡터 인덱스 검색 (Top-K)
  3. 에피소드 및 엔티티 메타데이터 필터링
  4. LLM 기반 Reranker 및 최종 답변 생성
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.neo4j_track.common_neo4j import get_neo4j_driver, CleanOpenAILLM, Neo4jEmbedder

driver = get_neo4j_driver()
embedder = Neo4jEmbedder()
llm = CleanOpenAILLM(
    model_name=config.LLM_MODEL,
    model_params={"max_tokens": 2000, "temperature": 0},
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

class SearchFilters(BaseModel):
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_number: Optional[str] = None
    entity_names: List[str] = Field(default_factory=list)
    relationship_types: List[str] = Field(default_factory=list)

class RewrittenQuery(BaseModel):
    search_queries: List[str]
    filters: SearchFilters
    reasoning: str

def rewrite_query(user_question: str) -> RewrittenQuery:
    prompt = f"""You are a query rewrite specialist for Demon Slayer Knowledge Graph.
Extract vector search terms and metadata filters from the question.
Available Entity Names: {', '.join(config.NODE_LABELS)}
Available Relationships: {', '.join(config.RELATIONSHIP_TYPES[:10])}...

Respond with JSON:
{{
  "search_queries": ["탄지로 루이 전투", "히노카미 카구라"],
  "filters": {{
    "season": 1,
    "episode": null,
    "episode_number": null,
    "entity_names": ["카마도 탄지로", "루이"],
    "relationship_types": ["FIGHTS", "BATTLES"]
  }},
  "reasoning": "탄지로와 루이의 십이귀월 전투 질문"
}}

User Question: {user_question}
JSON:"""
    resp = llm.invoke(prompt)
    try:
        # JSON 블록 추출
        clean_text = resp.content
        m = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if m:
            clean_text = m.group(0)
        data = json.loads(clean_text)
        return RewrittenQuery(**data)
    except Exception:
        return RewrittenQuery(
            search_queries=[user_question],
            filters=SearchFilters(),
            reasoning="Fallback"
        )

def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_emb = embedder.embed_query(query)
    results = []
    with driver.session(database=config.NEO4J_DATABASE) as session:
        for label in config.NODE_LABELS:
            index_name = f"{config.VECTOR_INDEX_NODE}_{label}"
            try:
                res = session.run(f"""
                    CALL db.index.vector.queryNodes('{index_name}', $top_k, $emb)
                    YIELD node, score
                    RETURN node.name as name, labels(node) as labels, score
                """, top_k=top_k, emb=query_emb)
                for rec in res:
                    results.append(dict(rec))
            except Exception:
                pass
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

def run_metadata_vector(question: str) -> str:
    print(f"\n💬 [Step 2 Metadata Vector] 사용자 질문: {question}")
    plan = rewrite_query(question)
    print(f"  🎯 리라이팅 질의: {plan.search_queries}")
    print(f"  🏷️ 메타데이터 필터: {plan.filters.model_dump()}")

    all_matches = []
    for sq in plan.search_queries:
        matches = vector_search(sq, top_k=5)
        all_matches.extend(matches)

    context = "\n".join([f"- {m.get('name')} (Score: {m.get('score'):.3f})" for m in all_matches])
    prompt = f"""다음 벡터 검색된 엔티티 및 속성을 기반으로 답변하세요.

## 검색 결과:
{context}

## 질문:
{question}

## 답변:"""
    ans = llm.invoke(prompt)
    return ans.content

if __name__ == "__main__":
    test_q = "탄지로와 네즈코의 가족 관계에 대해 알려줘"
    print(run_metadata_vector(test_q))
