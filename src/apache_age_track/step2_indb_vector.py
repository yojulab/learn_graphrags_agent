"""
[Apache AGE Track Step 2] PostgreSQL In-Database 코사인 유사도 벡터 검색 실습
(src/03_apache_age_track/3_step2_indb_vector.py)

- 실습 내용:
  1. 사용자 질의(자연어) ➔ bge-m3 1024차원 임베딩 생성
  2. PostgreSQL 사용자 정의 함수 `cosine_similarity(a float8[], b float8[])` 활용
  3. `demon_slayer_node_embeddings` 및 `demon_slayer_rel_embeddings` 관계형 테이블 인덱스 순위 검색
  4. 시드 노드/관계 추출 및 RDBMS 메타데이터 결합
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.apache_age_track.common_age import get_age_connection, AGEEmbedder

embedder = AGEEmbedder()

def search_nodes_by_vector(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """In-Database 코사인 유사도로 시드 노드 검색"""
    emb = embedder.embed_query(query_text)
    conn = get_age_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, label, name, role,
                   cosine_similarity(embedding, %s::float8[]) as similarity
            FROM demon_slayer_node_embeddings
            ORDER BY similarity DESC
            LIMIT %s;
            """, (emb, top_k))
            rows = cur.fetchall()
            return [
                {"id": r[0], "label": r[1], "name": r[2], "role": r[3], "similarity": float(r[4])}
                for r in rows
            ]
    finally:
        conn.close()

def search_relationships_by_vector(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """In-Database 코사인 유사도로 관련 관계 검색"""
    emb = embedder.embed_query(query_text)
    conn = get_age_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT start_id, end_id, type, episode_number, context, outcome, technique,
                   cosine_similarity(embedding, %s::float8[]) as similarity
            FROM demon_slayer_rel_embeddings
            ORDER BY similarity DESC
            LIMIT %s;
            """, (emb, top_k))
            rows = cur.fetchall()
            return [
                {
                    "start_id": r[0], "end_id": r[1], "type": r[2],
                    "episode_number": r[3], "context": r[4], "outcome": r[5],
                    "technique": r[6], "similarity": float(r[7])
                }
                for r in rows
            ]
    finally:
        conn.close()

def run_indb_vector_demo(query: str):
    print("=" * 80)
    print(f"🚀 [Step 2 In-DB Vector Search] 질의: '{query}'")
    print("=" * 80)

    print("\n1️⃣ 노드(인물) 유사도 Top-3 검색 결과:")
    nodes = search_nodes_by_vector(query, top_k=3)
    for n in nodes:
        print(f"  👤 [{n['label']}] {n['name']} (유사도: {n['similarity']:.4f})")

    print("\n2️⃣ 관계(사건/전투) 유사도 Top-3 검색 결과:")
    rels = search_relationships_by_vector(query, top_k=3)
    for r in rels:
        print(f"  ⚔️ [S1E{r['episode_number']}] {r['type']} (유사도: {r['similarity']:.4f}) - {r['context'][:60]}...")

    print("=" * 80)

if __name__ == "__main__":
    run_indb_vector_demo("루이와의 거미산 전투와 히노카미 카구라")
