"""
[Apache AGE Track Step 3] Apache AGE Multi-Model 하이브리드 GraphRAG 에이전트 실습
(src/03_apache_age_track/4_step3_hybrid_agent.py)

- 4단계 하네스 아키텍처:
  1. Stage 1: 질문 의도 분류기 (Query Intent Classifier & Named Entity Extractor)
  2. Stage 2: PostgreSQL In-Database Cosine Similarity 벡터 검색 (Seed Node/Rel Retrieval)
  3. Stage 3: openCypher in SQL 질의 생성 & 1~2 hop 동적 서브그래프 확장
  4. Stage 4: Think Tag 정제 및 한국어 구조화 답변 합성기 (Anti-Hallucination Guardrail)
"""

import sys
import re
import json
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import psycopg2
import openai

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.apache_age_track.common_age import get_age_connection, parse_agtype, AGEEmbedder

embedder = AGEEmbedder()
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL,
    timeout=300.0
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
        entities["entity"] = matched[0]
        return QueryType.SINGLE_ENTITY, entities

    return QueryType.GENERAL, entities

def search_seed_nodes(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    emb = embedder.embed_query(query_text)
    conn = get_age_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, label, name, cosine_similarity(embedding, %s::float8[]) as sim
            FROM demon_slayer_node_embeddings
            ORDER BY sim DESC
            LIMIT %s;
            """, (emb, top_k))
            return [{"id": r[0], "label": r[1], "name": r[2], "score": float(r[3])} for r in cur.fetchall()]
    finally:
        conn.close()

def expand_age_subgraph(seed_names: List[str]) -> List[Dict[str, Any]]:
    if not seed_names:
        return []
    conn = get_age_connection()
    records = []
    try:
        with conn.cursor() as cur:
            for s_name in seed_names:
                query = f"""
                SELECT * FROM cypher('demon_slayer', $$
                    MATCH (a {{name: '{s_name}'}})-[r]-(b)
                    RETURN a.name, type(r), b.name, r.episode_number, r.context, r.outcome, r.technique
                    LIMIT 20
                $$) as (src agtype, rel agtype, tgt agtype, ep agtype, ctx agtype, outcome agtype, tech agtype);
                """
                try:
                    cur.execute(query)
                    for r in cur.fetchall():
                        records.append({
                            "source": parse_agtype(r[0]),
                            "rel_type": parse_agtype(r[1]),
                            "target": parse_agtype(r[2]),
                            "episode": parse_agtype(r[3]),
                            "context": parse_agtype(r[4]),
                            "outcome": parse_agtype(r[5]),
                            "technique": parse_agtype(r[6]),
                        })
                except Exception as e:
                    pass
    finally:
        conn.close()
    return records

def execute_age_hybrid(question: str) -> str:
    print(f"\n🚀 [Apache AGE Track Step 3] 하이브리드 질의 실행: '{question}'")
    q_type, entities = classify_query_type(question)
    print(f"  🧭 의도: {q_type.value}, 추출 개체: {entities}")

    # 1. In-DB 벡터 시드 검색
    seeds = search_seed_nodes(question, top_k=3)
    seed_names = [s["name"] for s in seeds]
    print(f"  🌱 벡터 시드: {seed_names}")

    # 2. openCypher 1~2 Hop 서브그래프 확장
    expanded = expand_age_subgraph(seed_names)
    print(f"  🌿 AGE 확장 관계: {len(expanded)}건 확보")

    # 3. 관계 질의 보강
    if q_type == QueryType.RELATIONSHIP and "entity_a" in entities and "entity_b" in entities:
        conn = get_age_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT * FROM cypher('demon_slayer', $$
                    MATCH (a {{name: '{entities['entity_a']}'}})-[r]-(b {{name: '{entities['entity_b']}'}})
                    RETURN a.name, type(r), b.name, r.episode_number, r.context, r.outcome, r.technique
                $$) as (src agtype, rel agtype, tgt agtype, ep agtype, ctx agtype, outcome agtype, tech agtype);
                """)
                for r in cur.fetchall():
                    expanded.insert(0, {
                        "source": parse_agtype(r[0]),
                        "rel_type": parse_agtype(r[1]),
                        "target": parse_agtype(r[2]),
                        "episode": parse_agtype(r[3]),
                        "context": parse_agtype(r[4]),
                        "outcome": parse_agtype(r[5]),
                        "technique": parse_agtype(r[6]),
                    })
        finally:
            conn.close()

    if not expanded:
        return "지식 그래프 데이터상 확인되지 않는 정보입니다."

    # 4. 컨텍스트 구성
    context_lines = []
    seen = set()
    for f in expanded:
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

    # 5. LLM 답변 합성
    prompt = f"""당신은 귀멸의 칼날 지식그래프 전문 분석관입니다.
검색된 Apache AGE 지식 그래프 컨텍스트를 바탕으로 질문에 한국어로 명확하고 구체적인 개조식 답변을 작성하세요.
컨텍스트에 없는 내용은 추측하지 마세요.

## Apache AGE 지식그래프 컨텍스트:
{full_context}

## 사용자 질문:
{question}

## 구조화된 답변:"""

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    raw_answer = resp.choices[0].message.content or ""
    # think 태그 정제
    clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL)
    clean_answer = re.sub(r'<think>.*', '', clean_answer, flags=re.DOTALL)
    clean_answer = clean_answer.replace('</think>', '').strip()
    return clean_answer

if __name__ == "__main__":
    queries = [
        "카마도 탄지로가 루이와 싸운 에피소드는 몇 화이고 어떤 기술을 썼어?",
        "토미오카 기유는 시즌 1에서 어떤 활약을 했어?"
    ]
    for q in queries:
        print(execute_age_hybrid(q))
        print("-" * 80)
