"""
[Apache AGE Track Step 0] LPG 표준 데이터의 Apache AGE 및 PostgreSQL 멀티모델 적재
(src/03_apache_age_track/1_ingest_lpg_age.py)

- 엔터프라이즈 Multi-Model 아키텍처:
  1. Apache AGE Property Graph (ag_catalog) : demon_slayer 그래프 (정점 19개, 간선 310+개)
  2. PostgreSQL Relational Vector Tables : demon_slayer_node_embeddings, demon_slayer_rel_embeddings
  3. In-Database Cosine Similarity Function : 초고속 코사인 유사도 연산 함수 등록
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.apache_age_track.common_age import get_age_connection, escape_cypher_str, AGEEmbedder

def init_age_schema(cur):
    """PostgreSQL In-DB 코사인 유사도 함수 및 관계형 임베딩 테이블 스키마 초기화"""
    print("🛠️ [Apache AGE] In-Database 벡터 연산 함수 및 테이블 스키마 생성...")

    # 1. 초고속 코사인 유사도 함수 등록
    cur.execute("""
    CREATE OR REPLACE FUNCTION cosine_similarity(a float8[], b float8[])
    RETURNS float8 AS $$
    DECLARE
        dot_product float8 := 0;
        norm_a float8 := 0;
        norm_b float8 := 0;
        len int;
        i int;
    BEGIN
        len := array_length(a, 1);
        IF len IS NULL OR len != array_length(b, 1) THEN
            RETURN 0.0;
        END IF;

        FOR i IN 1..len LOOP
            dot_product := dot_product + (a[i] * b[i]);
            norm_a := norm_a + (a[i] * a[i]);
            norm_b := norm_b + (b[i] * b[i]);
        END LOOP;

        IF norm_a = 0 OR norm_b = 0 THEN
            RETURN 0.0;
        END IF;

        RETURN dot_product / (sqrt(norm_a) * sqrt(norm_b));
    END;
    $$ LANGUAGE plpgsql IMMUTABLE STRICT;
    """)

    # 2. 노드 임베딩 관계형 테이블
    cur.execute("""
    DROP TABLE IF EXISTS demon_slayer_node_embeddings CASCADE;
    CREATE TABLE demon_slayer_node_embeddings (
        id VARCHAR(50) PRIMARY KEY,
        label VARCHAR(50) NOT NULL,
        name VARCHAR(100) NOT NULL,
        role VARCHAR(100),
        embedding float8[] NOT NULL
    );
    """)

    # 3. 관계 임베딩 관계형 테이블
    cur.execute("""
    DROP TABLE IF EXISTS demon_slayer_rel_embeddings CASCADE;
    CREATE TABLE demon_slayer_rel_embeddings (
        id SERIAL PRIMARY KEY,
        start_id VARCHAR(50) NOT NULL,
        end_id VARCHAR(50) NOT NULL,
        type VARCHAR(100) NOT NULL,
        episode_number VARCHAR(20),
        context TEXT,
        outcome VARCHAR(100),
        technique VARCHAR(150),
        embedding float8[] NOT NULL
    );
    """)
    print("  ✅ 관계형 벡터 테이블 스키마 초기화 완료")

def ingest_lpg_to_age(data_path: str = "output/lpg_knowledge_graph.json"):
    kg_file = BASE_DIR / data_path
    if not kg_file.exists():
        kg_file = BASE_DIR / "output/knowledge_graph_v4.json"
    if not kg_file.exists():
        raise FileNotFoundError(f"LPG 지식그래프 파일을 찾을 수 없습니다: {kg_file}")

    print("=" * 80)
    print(f"🚀 [Apache AGE Track] Multi-Model 적재 파이프라인 시작: {kg_file}")
    print("=" * 80)

    with open(kg_file, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    nodes = kg_data.get("nodes", [])
    relationships = kg_data.get("relationships") or kg_data.get("edges", [])

    embedder = AGEEmbedder()
    conn = get_age_connection()

    try:
        with conn.cursor() as cur:
            # 1. 스키마 초기화
            init_age_schema(cur)

            # 2. Apache AGE 그래프 초기화 (멱등성 보장)
            graph_name = config.AGE_GRAPH_NAME
            cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s;", (graph_name,))
            if cur.fetchone()[0] > 0:
                cur.execute(f"SELECT drop_graph('{graph_name}', true);")
            cur.execute(f"SELECT create_graph('{graph_name}');")
            print(f"  ✅ Apache AGE '{graph_name}' 그래프 생성 완료")

            # 3. 정점(Vertex) 및 간선(Edge) 라벨 등록
            for lbl in config.NODE_LABELS:
                cur.execute(f"SELECT create_vlabel('{graph_name}', '{lbl}');")
            for rel_t in config.RELATIONSHIP_TYPES:
                try:
                    cur.execute(f"SELECT create_elabel('{graph_name}', '{rel_t}');")
                except Exception:
                    pass
            print("  ✅ Vertex/Edge 라벨 등록 완료")

            # 4. 정점(Vertex) 적재 및 임베딩 테이블 적재
            print(f"\n📦 정점 {len(nodes)}개 적재 중...")
            node_texts = [f"{n.get('label')}: {n.get('name')}" for n in nodes]
            node_embeddings = embedder.embed_documents(node_texts)

            node_records = []
            for node, emb in zip(nodes, node_embeddings):
                nid = node.get("id")
                label = node.get("label", "인간")
                name = node.get("name")
                role = node.get("properties", {}).get("role", "")

                # openCypher 정점 생성
                cur.execute(f"""
                SELECT * FROM cypher('{graph_name}', $$
                    CREATE (:{label} {{id: '{nid}', name: '{escape_cypher_str(name)}'}})
                $$) as (v agtype);
                """)

                node_records.append((nid, label, name, role, emb))

            # PostgreSQL 관계형 벡터 테이블에 벌크 삽입
            execute_batch(cur, """
                INSERT INTO demon_slayer_node_embeddings (id, label, name, role, embedding)
                VALUES (%s, %s, %s, %s, %s)
            """, node_records)
            print("  ✅ 정점 및 노드 임베딩 적재 완료")

            # 5. 간선(Edge) 적재 및 임베딩 테이블 적재
            print(f"\n🔗 간선 {len(relationships)}개 적재 중...")
            node_name_map = {n.get("id"): n.get("name") for n in nodes}
            rel_texts = []
            for r in relationships:
                s_id = r.get("start_node_id") or r.get("start_id")
                e_id = r.get("end_node_id") or r.get("end_id")
                s_name = node_name_map.get(s_id, s_id)
                e_name = node_name_map.get(e_id, e_id)
                ctx = r.get("properties", {}).get("context", "")
                rel_texts.append(f"{r.get('type')} - {s_name} → {e_name}: {ctx}")

            rel_embeddings = embedder.embed_documents(rel_texts)

            rel_records = []
            for rel, emb in zip(tqdm(relationships, desc="AGE 간선 적재"), rel_embeddings):
                s_id = rel.get("start_node_id") or rel.get("start_id")
                e_id = rel.get("end_node_id") or rel.get("end_id")
                rel_type = rel.get("type", "RELATED_TO").strip()
                props = rel.get("properties", {})
                ep = props.get("episode_number", "")
                ctx = props.get("context", "")
                outcome = props.get("outcome", "")
                tech = props.get("technique", "")

                # openCypher 간선 생성
                prop_str = f"episode_number: '{escape_cypher_str(ep)}', context: '{escape_cypher_str(ctx)}', outcome: '{escape_cypher_str(outcome)}', technique: '{escape_cypher_str(tech)}'"
                cur.execute(f"""
                SELECT * FROM cypher('{graph_name}', $$
                    MATCH (a {{id: '{s_id}'}}), (b {{id: '{e_id}'}})
                    CREATE (a)-[:{rel_type} {{{prop_str}}}]->(b)
                $$) as (e agtype);
                """)

                rel_records.append((s_id, e_id, rel_type, ep, ctx, outcome, tech, emb))

            # PostgreSQL 관계형 테이블에 벌크 삽입
            execute_batch(cur, """
                INSERT INTO demon_slayer_rel_embeddings 
                (start_id, end_id, type, episode_number, context, outcome, technique, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, rel_records)
            print("  ✅ 간선 및 관계 임베딩 적재 완료")

        print("=" * 80)
        print("✨ [Apache AGE Track] Multi-Model LPG 적재 성공 완료!")
        print("=" * 80)

    finally:
        conn.close()

if __name__ == "__main__":
    ingest_lpg_to_age()
