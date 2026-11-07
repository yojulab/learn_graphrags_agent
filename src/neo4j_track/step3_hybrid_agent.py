"""
[Neo4j Track Step 3] 하이브리드 GraphRAG 프로덕션 에이전트 실습
(src/02_neo4j_track/4_step3_hybrid_agent.py)

- 4단계 하네스 파이프라인:
  1. 질문 유형 분류기 (Query Intent Classifier & Named Entity Extractor)
  2. Neo4j HNSW 코사인 벡터 검색 (시드 노드 추출)
  3. 시드 노드 기준 1~2 Hop 서브그래프 동적 순회 및 Cypher 템플릿 결합
  4. Think Tag 정제 및 한국어 개조식 답변 합성기
"""

import sys
import re
import json
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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

class QueryType(Enum):
    SINGLE_ENTITY = "single_entity"
    RELATIONSHIP = "relationship"
    EPISODE_SPECIFIC = "episode_specific"
    GENERAL = "general"

def classify_query_type(question: str) -> Tuple[QueryType, Dict[str, Any]]:
    entities: Dict[str, Any] = {}
    q = question.strip()

    ep_match = re.search(r'(\d+)\s*(화|회|화차|episode|ep)', q, re.IGNORECASE)
    if ep_match:
        entities["episode_number"] = int(ep_match.group(1))
        return QueryType.EPISODE_SPECIFIC, entities

    all_known_names = [
        "카마도 탄지로", "카마도 네즈코", "토미오카 기유", "우로코다키 사콘지",
        "사비토", "마코모", "아가츠마 젠이츠", "하시비라 이노스케",
        "츠유리 카나오", "렌고쿠 쿄쥬로", "우부야시키 카가야", "코쵸우 시노부",
        "시나즈가와 사네미", "키부츠지 무잔", "스사마루", "야하바",
        "쿄우가이", "루이", "엔무"
    ]
    alias_map = {
        "탄지로": "카마도 탄지로", "네즈코": "카마도 네즈코", "기유": "토미오카 기유",
        "젠이츠": "아가츠마 젠이츠", "이노스케": "하시비라 이노스케", "무잔": "키부츠지 무잔"
    }

    found = set()
    for name in all_known_names:
        if name in q:
            found.add(name)
    if not found:
        for alias, full in alias_map.items():
            if alias in q:
                found.add(full)

    matched = list(found)
    if len(matched) >= 2:
        entities["entity_a"] = matched[0]
        entities["entity_b"] = matched[1]
        return QueryType.RELATIONSHIP, entities
    elif len(matched) == 1:
        entities["entity_name"] = matched[0]
        return QueryType.SINGLE_ENTITY, entities

    return QueryType.GENERAL, entities

def vector_seed_search(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    emb = embedder.embed_query(query_text)
    seeds = []
    with driver.session(database=config.NEO4J_DATABASE) as session:
        for label in config.NODE_LABELS:
            index_name = f"{config.VECTOR_INDEX_NODE}_{label}"
            try:
                res = session.run(f"""
                    CALL db.index.vector.queryNodes('{index_name}', $top_k, $emb)
                    YIELD node, score
                    RETURN node.id as id, node.name as name, labels(node) as labels, score
                """, top_k=top_k, emb=emb)
                for r in res:
                    seeds.append(dict(r))
            except Exception:
                pass
    return sorted(seeds, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

def expand_subgraph_from_seeds(seed_ids: List[str], max_depth: int = 2) -> List[Dict[str, Any]]:
    if not seed_ids:
        return []
    with driver.session(database=config.NEO4J_DATABASE) as session:
        res = session.run("""
            MATCH (seed) WHERE seed.id IN $seed_ids
            MATCH path = (seed)-[r*1..2]-(neighbor)
            UNWIND relationships(path) as rel
            RETURN DISTINCT
                startNode(rel).name as source,
                type(rel) as rel_type,
                endNode(rel).name as target,
                rel.episode_number as episode,
                rel.context as context,
                rel.outcome as outcome,
                rel.technique as technique
            LIMIT 40
        """, seed_ids=seed_ids)
        return [dict(r) for r in res]

def execute_hybrid_graphrag(question: str) -> str:
    print(f"\n🚀 [Step 3 Hybrid GraphRAG] 질문: {question}")
    q_type, entities = classify_query_type(question)
    print(f"  🧭 질문 의도 분류: {q_type.value}, 추출 개체: {entities}")

    # 1. 시드 벡터 검색
    seeds = vector_seed_search(question, top_k=3)
    seed_ids = [s["id"] for s in seeds]
    print(f"  🌱 벡터 시드 노드: {[s['name'] for s in seeds]}")

    # 2. 1~2 Hop 서브그래프 확장
    expanded_records = expand_subgraph_from_seeds(seed_ids, max_depth=config.GRAPH_EXPANSION_DEPTH)
    print(f"  🌿 그래프 확장 결과: {len(expanded_records)}개 관계 확보")

    # 3. 템플릿 Cypher 보조 질의
    template_records = []
    if q_type == QueryType.RELATIONSHIP and "entity_a" in entities and "entity_b" in entities:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            res = session.run("""
                MATCH (a {name: $ea})-[r]-(b {name: $eb})
                RETURN a.name as source, type(r) as rel_type, b.name as target,
                       r.episode_number as episode, r.context as context,
                       r.outcome as outcome, r.technique as technique
            """, ea=entities["entity_a"], eb=entities["entity_b"])
            template_records = [dict(r) for r in res]

    all_facts = template_records + expanded_records
    if not all_facts:
        return "지식 그래프 데이터상 확인되지 않는 정보입니다."

    # 4. 컨텍스트 포맷팅 및 중복 제거
    context_lines = []
    seen = set()
    for f in all_facts:
        key = (f.get("source"), f.get("rel_type"), f.get("target"), f.get("episode"))
        if key not in seen:
            seen.add(key)
            line = f"[{f.get('episode', 'N/A')}] {f.get('source')} --({f.get('rel_type')})--> {f.get('target')}"
            if f.get("technique"):
                line += f" | 기술: {f.get('technique')}"
            if f.get("outcome"):
                line += f" | 결과: {f.get('outcome')}"
            if f.get("context"):
                line += f" | 내용: {f.get('context')}"
            context_lines.append(line)

    full_context = "\n".join(context_lines[:25])

    # 5. 최종 답변 합성
    prompt = f"""당신은 귀멸의 칼날 지식그래프 전문 분석관입니다.
검색된 지식 그래프 컨텍스트를 기반으로 사용자의 질문에 한국어로 명확하고 구체적인 개조식 답변을 작성하세요.
컨텍스트에 없는 내용은 추측하지 마세요.

## 지식그래프 검증 컨텍스트:
{full_context}

## 사용자 질문:
{question}

## 구조화된 답변:"""

    resp = llm.invoke(prompt)
    return resp.content

if __name__ == "__main__":
    queries = [
        "카마도 탄지로가 루이와 싸운 에피소드는 몇 화이고 어떤 기술을 썼어?",
        "토미오카 기유와 탄지로의 관계는?"
    ]
    for q in queries:
        print(execute_hybrid_graphrag(q))
        print("-" * 80)
