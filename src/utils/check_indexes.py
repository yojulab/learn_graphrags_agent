"""
[Index & Schema Inspector] 듀얼 그래프 DB (Neo4j & Apache AGE) 인덱스 및 그래프 스키마 상태 점검
(src/utils/check_indexes.py)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

def check_neo4j_indexes():
    print("=" * 80)
    print("🟢 [Neo4j] HNSW 코사인 벡터 인덱스 현황")
    print("=" * 80)
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run("SHOW VECTOR INDEXES")
            print(f"{'Index Name':<42} | {'Entity':<12} | {'Labels/Types':<20}")
            print("-" * 80)
            count = 0
            for record in result:
                count += 1
                name = record.get('name', 'N/A')
                entity = record.get('entityType', 'N/A')
                lbls = str(record.get('labelsOrTypes', 'N/A'))
                print(f"{name:<42} | {entity:<12} | {lbls:<20}")
            print(f"\n총 {count}개의 Neo4j 벡터 인덱스가 등록되어 있습니다.")

            # 노드 및 관계 개수
            n_cnt = session.run("MATCH (n) RETURN count(n) as cnt").single()["cnt"]
            r_cnt = session.run("MATCH ()-[r]->() RETURN count(r) as cnt").single()["cnt"]
            print(f"📊 정점(Node): {n_cnt}개, 간선(Relationship): {r_cnt}개")
        driver.close()
    except Exception as e:
        print(f"❌ Neo4j 점검 실패: {e}")

def check_apache_age_schema():
    print("\n" + "=" * 80)
    print("🐘 [Apache AGE & PostgreSQL] 그래프 카탈로그 및 관계형 벡터 테이블 현황")
    print("=" * 80)
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB
        )
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute('SET search_path = ag_catalog, "$user", public;')

            # 1. 그래프 확인
            cur.execute("SELECT name FROM ag_graph;")
            graphs = [r[0] for r in cur.fetchall()]
            print(f"📈 생성된 그래프 목록: {graphs}")

            # 2. 정점 및 간선 라벨 확인
            if config.AGE_GRAPH_NAME in graphs:
                cur.execute(f"SELECT name, kind FROM ag_label WHERE graph = (SELECT graphid FROM ag_graph WHERE name = '{config.AGE_GRAPH_NAME}');")
                labels = cur.fetchall()
                v_labels = [l[0] for l in labels if l[1] == 'v']
                e_labels = [l[0] for l in labels if l[1] == 'e']
                print(f"  🏷️ 정점 라벨 ({len(v_labels)}종): {v_labels}")
                print(f"  🔗 간선 라벨 ({len(e_labels)}종): {e_labels[:8]}... (총 {len(e_labels)}개)")

            # 3. 관계형 벡터 테이블 레코드 확인
            cur.execute("""
            SELECT 
                (SELECT count(*) FROM demon_slayer_node_embeddings) as nodes,
                (SELECT count(*) FROM demon_slayer_rel_embeddings) as rels;
            """)
            counts = cur.fetchone()
            print(f"📊 PostgreSQL 임베딩 테이블: 노드 {counts[0]}건, 관계 {counts[1]}건")

            # 4. 코사인 유사도 함수 확인
            cur.execute("SELECT proname FROM pg_proc WHERE proname = 'cosine_similarity';")
            fn = cur.fetchone()
            print(f"⚡ In-Database 코사인 유사도 함수: {'등록됨 (정상)' if fn else '미등록'}")
        conn.close()
    except Exception as e:
        print(f"❌ Apache AGE 점검 실패: {e}")

if __name__ == "__main__":
    check_neo4j_indexes()
    check_apache_age_schema()
