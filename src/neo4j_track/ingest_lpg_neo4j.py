"""
[Neo4j Track Step 0] LPG 표준 데이터의 Neo4j 적재 및 24종 HNSW 벡터 인덱스 생성
(src/02_neo4j_track/1_ingest_lpg_neo4j.py)

- 멱등성 보장 (MATCH (n) DETACH DELETE n)
- 정점 MERGE (라벨: :인간, :도깨비) + 노드 속성
- 간선 CREATE (24종 표준 관계 동사) + 간선 속성
- bge-m3 1024차원 임베딩 생성 및 Neo4j HNSW 코사인 벡터 인덱스 구축
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.neo4j_track.common_neo4j import get_neo4j_driver, Neo4jEmbedder

def create_vector_indexes(driver, database: str = config.NEO4J_DATABASE):
    """노드 및 관계 코사인 벡터 인덱스 자동 생성"""
    print("🔍 [Neo4j] HNSW 벡터 인덱스 구축 중...")
    with driver.session(database=database) as session:
        # 1. 노드 벡터 인덱스 (:인간, :도깨비)
        for label in config.NODE_LABELS:
            index_name = f"{config.VECTOR_INDEX_NODE}_{label}"
            session.run(f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {config.EMBEDDING_DIMENSION},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            print(f"  ✅ 노드 벡터 인덱스 확인: {index_name}")

        # 2. 24개 관계 벡터 인덱스
        for rel_type in config.RELATIONSHIP_TYPES:
            index_name = f"{config.VECTOR_INDEX_RELATIONSHIP_PREFIX}_{rel_type}"
            try:
                session.run(f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR ()-[r:{rel_type}]-()
                    ON r.embedding
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {config.EMBEDDING_DIMENSION},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
            except Exception as e:
                pass
        print("  ✅ 24종 관계 벡터 인덱스 등록 완료")

def ingest_lpg_to_neo4j(data_path: str = "output/lpg_knowledge_graph.json"):
    kg_file = BASE_DIR / data_path
    if not kg_file.exists():
        kg_file = BASE_DIR / "output/knowledge_graph_v3.json"
    if not kg_file.exists():
        raise FileNotFoundError(f"LPG 지식그래프 파일을 찾을 수 없습니다: {kg_file}")

    print("=" * 80)
    print(f"🚀 [Neo4j Track] LPG 적재 파이프라인 시작: {kg_file}")
    print("=" * 80)

    with open(kg_file, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    nodes = kg_data.get("nodes", [])
    relationships = kg_data.get("relationships") or kg_data.get("edges", [])

    embedder = Neo4jEmbedder()
    driver = get_neo4j_driver()

    try:
        # 1. 멱등 초기화
        print("🧹 기존 Neo4j 데이터 초기화 (DETACH DELETE)...")
        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")

        # 2. 벡터 인덱스 생성
        create_vector_indexes(driver)

        # 3. 노드 임베딩 및 적재
        print(f"\n📦 노드 {len(nodes)}개 임베딩 생성 및 적재...")
        node_texts = [f"{n.get('label')}: {n.get('name')}" for n in nodes]
        node_embeddings = embedder.embed_documents(node_texts)

        with driver.session(database=config.NEO4J_DATABASE) as session:
            for node, emb in zip(nodes, node_embeddings):
                label = node.get("label", "인간")
                props = node.get("properties", {}).copy()
                props["name"] = node.get("name")
                props["id"] = node.get("id")
                props["embedding"] = emb

                session.run(f"""
                    MERGE (n:`{label}` {{id: $id}})
                    SET n += $props
                """, id=node.get("id"), props=props)
            print("  ✅ 노드 적재 완료")

        # 4. 관계 임베딩 및 적재
        print(f"\n🔗 관계 {len(relationships)}개 임베딩 생성 및 적재...")
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

        with driver.session(database=config.NEO4J_DATABASE) as session:
            for rel, emb in zip(tqdm(relationships, desc="적재 중"), rel_embeddings):
                s_id = rel.get("start_node_id") or rel.get("start_id")
                e_id = rel.get("end_node_id") or rel.get("end_id")
                rel_type = rel.get("type", "RELATED_TO").strip()
                props = rel.get("properties", {}).copy()
                props["embedding"] = emb

                session.run(f"""
                    MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
                    CREATE (a)-[r:`{rel_type}`]->(b)
                    SET r += $props
                """, start_id=s_id, end_id=e_id, props=props)
            print("  ✅ 관계 적재 완료")

        print("=" * 80)
        print("✨ [Neo4j Track] LPG 적재 및 인덱싱 성공 완료!")
        print("=" * 80)

    finally:
        driver.close()

if __name__ == "__main__":
    ingest_lpg_to_neo4j()
