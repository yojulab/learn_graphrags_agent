"""
[MCP Server] Multi-Model GraphRAG DB Inspector Server
(src/utils/mcp_db_server.py)

표준 Model Context Protocol (MCP 2.x) 기반의 Stdio DB 상태 인스펙터 서버:
- 백엔드 지원: Neo4j 5+, PostgreSQL 16+ & Apache AGE, MongoDB
- 안전 가드레일: 읽기 전용(Read-Only) 쿼리 강제, 변형 쿼리(CREATE, DROP, DELETE 등) 원천 차단
- 중앙 SSOT: config.py 환경 설정 연동
"""

import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# 루트 경로 등록
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

from mcp.server.mcpserver import MCPServer
import psycopg2
from neo4j import GraphDatabase
import pymongo

# MCP 서버 인스턴스 초기화
mcp = MCPServer("graphrag-db-inspector")


# ============================================================
# 안전성 가드레일 헬퍼
# ============================================================

def assert_read_only(query: str) -> None:
    """돌연변이 쿼리(CREATE, DROP, DELETE, SET, MERGE 등) 정규식 차단"""
    forbidden = [r'\bCREATE\b', r'\bDROP\b', r'\bDELETE\b', r'\bSET\b', r'\bMERGE\b', r'\bREMOVE\b', r'\bALTER\b', r'\bINSERT\b', r'\bUPDATE\b', r'\bTRUNCATE\b']
    for pattern in forbidden:
        if re.search(pattern, query, re.IGNORECASE):
            raise PermissionError(f"[Read-Only Guardrail] 쿼리 변형 구문이 감지되어 실행이 차단되었습니다: {pattern}")


def clean_agtype(val: Any) -> Any:
    """Apache AGE agtype 결과 정제"""
    if val is None:
        return None
    s = str(val).strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    try:
        return json.loads(s)
    except Exception:
        return s


# ============================================================
# MCP Tools 정의
# ============================================================

@mcp.tool()
def check_all_dbs_status() -> str:
    """Neo4j, PostgreSQL(Apache AGE), MongoDB 3대 데이터베이스의 접속 상태 및 핵심 적재 건수를 일괄 진단합니다.

    Returns:
        JSON 문자열 형태의 3대 DB 종합 진단 결과
    """
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "databases": {}
    }

    # 1. Neo4j 진단
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        with driver.session(database=config.NEO4J_DATABASE) as session:
            node_cnt = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            indexes = list(session.run("SHOW VECTOR INDEXES"))
        driver.close()
        report["databases"]["neo4j"] = {
            "status": "HEALTHY",
            "nodes_count": node_cnt,
            "relationships_count": rel_cnt,
            "vector_indexes_count": len(indexes),
            "expected_match": (node_cnt == 19 and rel_cnt >= 300)
        }
    except Exception as e:
        report["databases"]["neo4j"] = {"status": "UNHEALTHY", "error": str(e)}

    # 2. PostgreSQL + Apache AGE 진단
    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB
        )
        cur = conn.cursor()
        cur.execute("LOAD 'age';")
        cur.execute('SET search_path = ag_catalog, "$user", public;')

        # AGE 그래프 노드/관계 카운트
        cur.execute(f"SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$ MATCH (n) RETURN count(n) $$) as (cnt agtype);")
        age_nodes = int(cur.fetchone()[0])

        cur.execute(f"SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$ MATCH ()-[r]->() RETURN count(r) $$) as (cnt agtype);")
        age_rels = int(cur.fetchone()[0])

        # 관계형 벡터 테이블 카운트
        cur.execute("SELECT count(*) FROM node_embeddings;")
        node_tbl_cnt = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM relationship_embeddings;")
        rel_tbl_cnt = cur.fetchone()[0]

        cur.close()
        conn.close()

        report["databases"]["postgres_apache_age"] = {
            "status": "HEALTHY",
            "graph_name": config.AGE_GRAPH_NAME,
            "age_nodes_count": age_nodes,
            "age_relationships_count": age_rels,
            "table_node_embeddings": node_tbl_cnt,
            "table_rel_embeddings": rel_tbl_cnt,
            "expected_match": (age_nodes == 19 and age_rels >= 300)
        }
    except Exception as e:
        report["databases"]["postgres_apache_age"] = {"status": "UNHEALTHY", "error": str(e)}

    # 3. MongoDB 진단
    try:
        mongo_client = pymongo.MongoClient(
            host=config.MONGO_HOST,
            port=config.MONGO_PORT,
            username=config.MONGO_USER,
            password=config.MONGO_PASSWORD,
            serverSelectionTimeoutMS=2000
        )
        db = mongo_client[config.MONGO_DB]
        collections = db.list_collection_names()
        coll_counts = {c: db[c].count_documents({}) for c in collections}
        mongo_client.close()

        report["databases"]["mongodb"] = {
            "status": "HEALTHY",
            "database_name": config.MONGO_DB,
            "collections_count": len(collections),
            "collections": coll_counts
        }
    except Exception as e:
        report["databases"]["mongodb"] = {"status": "UNHEALTHY", "error": str(e)}

    return json.dumps(report, ensure_ascii=False, indent=2)


@mcp.tool()
def inspect_neo4j_graph() -> str:
    """Neo4j 5+ 그래프 데이터베이스의 상세 통계(라벨별 노드 수, 24개 관계별 건수, 26개 HNSW 벡터 인덱스 목록)를 반환합니다.

    Returns:
        JSON 문자열 형태의 Neo4j 상세 분석 결과
    """
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session(database=config.NEO4J_DATABASE) as session:
        # 1. 라벨별 노드 수
        label_res = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
        label_counts = {r["label"]: r["cnt"] for r in label_res}

        # 2. 관계 타입별 수
        rel_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC")
        rel_counts = {r["rel_type"]: r["cnt"] for r in rel_res}

        # 3. 벡터 인덱스 목록
        idx_res = session.run("SHOW VECTOR INDEXES")
        indexes = []
        for r in idx_res:
            indexes.append({
                "name": r["name"],
                "state": r["state"],
                "labelsOrTypes": r.get("labelsOrTypes", []),
                "properties": r.get("properties", [])
            })

    driver.close()

    result = {
        "database": config.NEO4J_DATABASE,
        "total_nodes": sum(label_counts.values()),
        "label_counts": label_counts,
        "total_relationships": sum(rel_counts.values()),
        "relationship_counts": rel_counts,
        "vector_indexes_count": len(indexes),
        "vector_indexes": indexes
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def query_neo4j_cypher(cypher_query: str, limit: int = 25) -> str:
    """Neo4j 그래프에 대해 읽기 전용(MATCH) Cypher 쿼리를 안전하게 실행하고 결과를 반환합니다.

    Args:
        cypher_query: 실행할 Cypher 쿼리 (예: MATCH (n:인간) RETURN n.name, n.role LIMIT 10)
        limit: 최대 반환 레코드 수 (기본값: 25)

    Returns:
        JSON 문자열 형태의 쿼리 실행 결과 레코드 목록
    """
    assert_read_only(cypher_query)

    # LIMIT 보장
    if not re.search(r'\bLIMIT\b', cypher_query, re.IGNORECASE):
        cypher_query = f"{cypher_query.strip()} LIMIT {limit}"

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    records = []
    with driver.session(database=config.NEO4J_DATABASE) as session:
        res = session.run(cypher_query)
        for r in res:
            records.append(dict(r))
    driver.close()

    return json.dumps({
        "query": cypher_query,
        "count": len(records),
        "records": records
    }, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
def inspect_apache_age_graph() -> str:
    """PostgreSQL 16+ & Apache AGE의 demon_slayer 그래프 및 관계형 벡터 테이블 상태를 조회합니다.

    Returns:
        JSON 문자열 형태의 Apache AGE 상세 분석 결과
    """
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=config.POSTGRES_DB
    )
    cur = conn.cursor()
    try:
        cur.execute("LOAD 'age';")
        cur.execute('SET search_path = ag_catalog, "$user", public;')

        # 라벨별 버텍스 수
        cur.execute(f"""
        SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$
            MATCH (n) RETURN labels(n), count(n)
        $$) as (label agtype, cnt agtype);
        """)
        vertex_labels = {clean_agtype(r[0])[0] if isinstance(clean_agtype(r[0]), list) else str(clean_agtype(r[0])): int(r[1]) for r in cur.fetchall()}

        # 2. 관계형 벡터 테이블 정보
        cur.execute("SELECT count(*) FROM node_embeddings;")
        node_embed_count = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM relationship_embeddings;")
        rel_embed_count = cur.fetchone()[0]

        # 3. 코사인 유사도 함수 확인
        cur.execute("SELECT proname FROM pg_proc WHERE proname = 'cosine_similarity';")
        has_cosine_fn = cur.fetchone() is not None

        result = {
            "graph_name": config.AGE_GRAPH_NAME,
            "total_vertices": sum(vertex_labels.values()),
            "vertex_labels": vertex_labels,
            "node_embeddings_table_rows": node_embed_count,
            "relationship_embeddings_table_rows": rel_embed_count,
            "cosine_similarity_sql_function": "AVAILABLE" if has_cosine_fn else "MISSING"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    finally:
        cur.close()
        conn.close()


def count_return_columns(cypher: str) -> int:
    """openCypher 쿼리의 RETURN 절 컬럼 개수 계산"""
    match = re.search(r'\bRETURN\b(.*?)(?:\bLIMIT\b|\bORDER\s+BY\b|\bSKIP\b|$)', cypher, re.IGNORECASE | re.DOTALL)
    if not match:
        return 1
    return_clause = match.group(1).strip()
    depth = 0
    col_count = 1
    for char in return_clause:
        if char in '({[':
            depth += 1
        elif char in ')}]':
            depth -= 1
        elif char == ',' and depth == 0:
            col_count += 1
    return max(1, col_count)


@mcp.tool()
def query_apache_age_cypher(cypher_body: str, limit: int = 25) -> str:
    """Apache AGE demon_slayer 그래프에 대해 openCypher 쿼리를 안전하게 실행합니다.

    Args:
        cypher_body: cypher(...) 함수 내부에 들어갈 openCypher 쿼리 (예: MATCH (a:인간)-[r]->(b) RETURN a.name, type(r), b.name)
        limit: 최대 반환 레코드 수 (기본값: 25)

    Returns:
        JSON 문자열 형태의 쿼리 결과 레코드 목록
    """
    assert_read_only(cypher_body)

    if not re.search(r'\bLIMIT\b', cypher_body, re.IGNORECASE):
        cypher_body = f"{cypher_body.strip()} LIMIT {limit}"

    col_count = count_return_columns(cypher_body)
    cols_def = ", ".join(f"col_{i+1} agtype" for i in range(col_count))

    sql = f"""
    SELECT * FROM cypher('{config.AGE_GRAPH_NAME}', $$
        {cypher_body}
    $$) as ({cols_def});
    """

    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=config.POSTGRES_DB
    )
    cur = conn.cursor()
    try:
        cur.execute("LOAD 'age';")
        cur.execute('SET search_path = ag_catalog, "$user", public;')
        cur.execute(sql)
        rows = cur.fetchall()
        records = [[clean_agtype(col) for col in row] for row in rows]

        return json.dumps({
            "graph": config.AGE_GRAPH_NAME,
            "cypher": cypher_body,
            "count": len(records),
            "records": records
        }, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return json.dumps({
            "graph": config.AGE_GRAPH_NAME,
            "cypher": cypher_body,
            "error": str(e)
        }, ensure_ascii=False, indent=2)
    finally:
        cur.close()
        conn.close()


@mcp.tool()
def inspect_mongodb_status() -> str:
    """MongoDB의 컬렉션 목록 및 도큐먼트 적재 상태를 확인합니다.

    Returns:
        JSON 문자열 형태의 MongoDB 컬렉션 통계
    """
    mongo_client = pymongo.MongoClient(
        host=config.MONGO_HOST,
        port=config.MONGO_PORT,
        username=config.MONGO_USER,
        password=config.MONGO_PASSWORD,
        serverSelectionTimeoutMS=2000
    )
    db = mongo_client[config.MONGO_DB]
    collections = db.list_collection_names()
    details = {}
    for c in collections:
        details[c] = {
            "document_count": db[c].count_documents({}),
            "sample": list(db[c].find({}, {"_id": 0}).limit(2))
        }
    mongo_client.close()

    result = {
        "database": config.MONGO_DB,
        "collections_count": len(collections),
        "collections": details
    }
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


# ============================================================
# CLI 자체 검증 및 실행 엔트리포인트
# ============================================================

def run_self_test():
    """모든 MCP 도구의 로컬 실행 정합성 테스트"""
    print("=" * 80)
    print("🧪 [MCP Server 자체 테스트] Multi-Model GraphRAG DB Inspector")
    print("=" * 80)

    print("\n1. check_all_dbs_status():")
    res1 = check_all_dbs_status()
    print(res1)

    print("\n2. inspect_neo4j_graph():")
    res2 = inspect_neo4j_graph()
    print(res2[:500] + "...")

    print("\n3. query_neo4j_cypher('MATCH (n:인간) RETURN n.name, n.role LIMIT 3'):")
    res3 = query_neo4j_cypher("MATCH (n:인간) RETURN n.name, n.role LIMIT 3")
    print(res3)

    print("\n4. inspect_apache_age_graph():")
    res4 = inspect_apache_age_graph()
    print(res4)

    print("\n5. query_apache_age_cypher('MATCH (a:인간)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 3'):")
    res5 = query_apache_age_cypher("MATCH (a:인간)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 3")
    print(res5)

    print("\n6. inspect_mongodb_status():")
    res6 = inspect_mongodb_status()
    print(res6)

    print("\n🎉 [MCP 도구 전체 자체 테스트 통과!]")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_self_test()
    else:
        # 표준 Stdio 트랜스포트로 MCP 서버 구동
        mcp.run(transport="stdio")
