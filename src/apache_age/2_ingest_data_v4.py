"""
[v4] PostgreSQL 16+ & Apache AGE 1.5+ 기반 멱등 지식그래프 적재 스크립트
(src/v4/2_ingest_data_v4.py)

엔터프라이즈 Multi-Model 아키텍처:
1. Apache AGE Property Graph (ag_catalog) : demon_slayer 그래프 (노드 19개, 관계 310개)
2. PostgreSQL Relational Vector Tables : node_embeddings, relationship_embeddings (1024차원 bge-m3)
3. In-Database Cosine Similarity Function : 초고속 코사인 유사도 연산 함수 등록
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_batch
import openai
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

# OpenAI 클라이언트 초기화 (Ollama bge-m3 임베딩)
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

def escape_cypher_str(val: Any) -> str:
    """Cypher 쿼리 내 문자열 리터럴 안전 이스케이프"""
    if val is None:
        return ""
    s = str(val)
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", " ")


class AGEEmbedder:
    """1024차원 bge-m3 임베딩 생성기"""
    def __init__(self, model: str = config.EMBEDDING_MODEL):
        self.model = model

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                res = client.embeddings.create(model=self.model, input=batch)
                all_embeddings.extend([d.embedding for d in res.data])
            except Exception as e:
                print(f"⚠️ 임베딩 배치 {i//batch_size + 1} 실패: {e}")
                all_embeddings.extend([[0.0] * config.EMBEDDING_DIMENSION for _ in batch])
        return all_embeddings


def create_node_text(node: Dict[str, Any]) -> str:
    props = node.get("properties", {})
    role = props.get("role", "")
    return f"{node.get('label')}: {node.get('name')} {role}".strip()


def create_rel_text(rel: Dict[str, Any], node_name_map: Dict[str, str]) -> str:
    start_name = node_name_map.get(rel.get("start_id", ""), rel.get("start_id", ""))
    end_name = node_name_map.get(rel.get("end_id", ""), rel.get("end_id", ""))
    props = rel.get("properties", {})
    context = props.get("context", "")
    ep = props.get("episode_number", "")
    return f"{rel.get('type')} - {start_name} → {end_name}: S1E{ep} {context}".strip()


def init_database(cur) -> None:
    """AGE 확장 및 관계형 벡터 테이블, 코사인 유사도 함수 초기화"""
    print("🛠️  PostgreSQL 확장 및 Multi-Model 스키마 초기화 중...")
    cur.execute("LOAD 'age';")
    cur.execute('SET search_path = ag_catalog, "$user", public;')

    # 1. 코사인 유사도 SQL 함수 생성 (PostgreSQL In-Database Vector Search)
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
    $$ LANGUAGE plpgsql IMMUTABLE;
    """)

    # 2. 관계형 벡터 테이블 생성
    cur.execute("DROP TABLE IF EXISTS node_embeddings CASCADE;")
    cur.execute("""
    CREATE TABLE node_embeddings (
        id VARCHAR(64) PRIMARY KEY,
        label VARCHAR(64) NOT NULL,
        name VARCHAR(255) NOT NULL,
        text TEXT NOT NULL,
        embedding float8[] NOT NULL
    );
    """)

    cur.execute("DROP TABLE IF EXISTS relationship_embeddings CASCADE;")
    cur.execute("""
    CREATE TABLE relationship_embeddings (
        id SERIAL PRIMARY KEY,
        start_id VARCHAR(64) NOT NULL,
        end_id VARCHAR(64) NOT NULL,
        rel_type VARCHAR(64) NOT NULL,
        context TEXT,
        episode_number INT,
        text TEXT NOT NULL,
        embedding float8[] NOT NULL
    );
    """)
    print("✅ PostgreSQL 관계형 테이블 및 cosine_similarity 함수 준비 완료")


def reset_age_graph(cur, graph_name: str) -> None:
    """Apache AGE 그래프 멱등 초기화 및 라벨 등록"""
    print(f"\n🕸️  Apache AGE 그래프 '{graph_name}' 초기화 중...")
    cur.execute("SELECT * FROM ag_graph WHERE name = %s;", (graph_name,))
    if cur.fetchone():
        cur.execute(f"SELECT drop_graph('{graph_name}', cascade => true);")
        print(f"🗑️  기존 그래프 '{graph_name}' 초기화 완료")

    cur.execute(f"SELECT create_graph('{graph_name}');")
    print(f"✨ 신규 그래프 '{graph_name}' 생성 완료")

    # 정점 라벨(인간, 도깨비) 등록
    for label in config.NODE_LABELS:
        cur.execute(f"SELECT create_vlabel('{graph_name}', '{label}');")
    print(f"🏷️  정점 라벨 등록 완료: {config.NODE_LABELS}")

    # 간선 라벨(24종 표준 관계) 등록
    for rel_type in config.RELATIONSHIP_TYPES:
        cur.execute(f"SELECT create_elabel('{graph_name}', '{rel_type}');")
    print(f"🔗 간선 라벨 24종 등록 완료: {len(config.RELATIONSHIP_TYPES)}개")


def ingest_v4_pipeline() -> Dict[str, Any]:
    """v4 전체 데이터 적재 실행 파이프라인"""
    print("=" * 80)
    print("🚀 [v4.0] PostgreSQL 16+ & Apache AGE Multi-Model 지식그래프 적재 시작")
    print("=" * 80)
    start_time = time.time()

    # 1. 데이터 로드
    v4_kg_path = BASE_DIR / "output/knowledge_graph_v4.json"
    if not v4_kg_path.exists():
        import importlib
        prep_mod = importlib.import_module("src.apache_age.1_prepare_data_v4")
        prep_mod.prepare_data_v4()

    with open(v4_kg_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    nodes = kg_data.get("nodes", [])
    relationships = kg_data.get("relationships", [])
    node_name_map = {n["id"]: n["name"] for n in nodes}

    print(f"📦 적재 대상: 노드 {len(nodes)}개, 관계 {len(relationships)}개")

    # 2. 임베딩 생성 (1024차원 bge-m3)
    embedder = AGEEmbedder()
    print(f"\n🧠 노드 {len(nodes)}개 텍스트 임베딩 생성 중...")
    node_texts = [create_node_text(n) for n in nodes]
    node_embeddings = embedder.embed_documents(node_texts)
    print(f"✅ 노드 임베딩 완료: {len(node_embeddings)}건")

    print(f"\n🧠 관계 {len(relationships)}개 텍스트 임베딩 생성 중...")
    rel_texts = [create_rel_text(r, node_name_map) for r in relationships]
    rel_embeddings = embedder.embed_documents(rel_texts)
    print(f"✅ 관계 임베딩 완료: {len(rel_embeddings)}건")

    # 3. PostgreSQL 연결 및 DB 초기화
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=config.POSTGRES_DB
    )
    conn.autocommit = True
    cur = conn.cursor()

    try:
        init_database(cur)
        reset_age_graph(cur, config.AGE_GRAPH_NAME)

        # 4. Apache AGE 노드 생성 & 관계형 node_embeddings 테이블 적재
        print(f"\n🌱 Apache AGE 정점(Vertex) 생성 및 관계형 테이블 적재 중...")
        node_records_to_insert = []
        for i, node in enumerate(tqdm(nodes, desc="Ingesting Nodes")):
            node_id = escape_cypher_str(node["id"])
            label = node["label"]
            name = escape_cypher_str(node["name"])
            props = node.get("properties", {})
            role = escape_cypher_str(props.get("role", ""))
            first_ep = props.get("first_appearance") or 1

            # AGE Vertex 생성 (openCypher)
            cypher_sql = f"""
            SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$
                CREATE (v:{label} {{
                    id: '{node_id}',
                    name: '{name}',
                    role: '{role}',
                    first_appearance: {first_ep}
                }})
                RETURN v
            $$) as (v agtype);
            """
            cur.execute(cypher_sql)

            # Relational Vector Table 레코드
            node_records_to_insert.append((
                node["id"],
                label,
                node["name"],
                node_texts[i],
                node_embeddings[i]
            ))

        # 관계형 벡터 테이블 일괄 저장
        execute_batch(
            cur,
            "INSERT INTO node_embeddings (id, label, name, text, embedding) VALUES (%s, %s, %s, %s, %s);",
            node_records_to_insert
        )
        print("✅ 노드 버텍스 및 관계형 벡터 테이블 적재 완료")

        # 5. Apache AGE 간선(Edge) 생성 & 관계형 relationship_embeddings 테이블 적재
        print(f"\n🔗 Apache AGE 간선(Edge) 생성 및 관계형 테이블 적재 중...")
        rel_records_to_insert = []
        for i, rel in enumerate(tqdm(relationships, desc="Ingesting Relationships")):
            start_id = escape_cypher_str(rel["start_id"])
            end_id = escape_cypher_str(rel["end_id"])
            rel_type = rel["type"]
            props = rel.get("properties", {})
            raw_ep = props.get("episode_number") or 1
            if isinstance(raw_ep, int):
                ep_num = raw_ep
            else:
                ep_match = re.search(r'\d+', str(raw_ep))
                ep_num = int(ep_match.group()) if ep_match else 1
            season = int(props.get("season") or 1)
            context = escape_cypher_str(props.get("context", ""))

            # AGE Edge 생성 (openCypher)
            cypher_sql = f"""
            SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$
                MATCH (a {{id: '{start_id}'}}), (b {{id: '{end_id}'}})
                CREATE (a)-[r:{rel_type} {{
                    episode_number: {ep_num},
                    season: {season},
                    context: '{context}'
                }}]->(b)
                RETURN r
            $$) as (r agtype);
            """
            cur.execute(cypher_sql)

            rel_records_to_insert.append((
                rel["start_id"],
                rel["end_id"],
                rel_type,
                props.get("context", ""),
                ep_num,
                rel_texts[i],
                rel_embeddings[i]
            ))

        execute_batch(
            cur,
            """
            INSERT INTO relationship_embeddings 
            (start_id, end_id, rel_type, context, episode_number, text, embedding) 
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            rel_records_to_insert
        )
        print("✅ 관계 간선 및 관계형 벡터 테이블 적재 완료")

        # 6. 정합성 검증 (AGE & PostgreSQL)
        print("\n🔍 적재 정합성 검증 중...")
        cur.execute(f"""
        SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$
            MATCH (n) RETURN count(n)
        $$) as (cnt agtype);
        """)
        age_nodes = int(cur.fetchone()[0])

        cur.execute(f"""
        SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$
            MATCH ()-[r]->() RETURN count(r)
        $$) as (cnt agtype);
        """)
        age_rels = int(cur.fetchone()[0])

        cur.execute("SELECT count(*) FROM node_embeddings;")
        rel_table_nodes = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM relationship_embeddings;")
        rel_table_rels = cur.fetchone()[0]

        elapsed_time = time.time() - start_time
        print(f"  • AGE 그래프 노드 수 : {age_nodes}개 (기대: {len(nodes)})")
        print(f"  • AGE 그래프 관계 수 : {age_rels}개 (기대: {len(relationships)})")
        print(f"  • 벡터 테이블 노드 수 : {rel_table_nodes}개")
        print(f"  • 벡터 테이블 관계 수 : {rel_table_rels}개")
        print(f"⏱️  전체 적재 완료 소요 시간: {elapsed_time:.2f}s")

        assert age_nodes == len(nodes), f"AGE 노드 수 불일치: {age_nodes} != {len(nodes)}"
        assert age_rels == len(relationships), f"AGE 관계 수 불일치: {age_rels} != {len(relationships)}"
        assert rel_table_nodes == len(nodes), f"관계형 노드 수 불일치: {rel_table_nodes}"
        assert rel_table_rels == len(relationships), f"관계형 관계 수 불일치: {rel_table_rels}"

        print(f"\n🎉 [v4 적재 성공] Apache AGE 그래프 및 Multi-Model 벡터 테이블 적재가 완벽히 완료되었습니다!")
        print("=" * 80 + "\n")

        return {
            "status": "SUCCESS",
            "age_nodes": age_nodes,
            "age_relationships": age_rels,
            "node_embeddings": rel_table_nodes,
            "rel_embeddings": rel_table_rels,
            "duration": f"{elapsed_time:.2f}s"
        }

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    ingest_v4_pipeline()
