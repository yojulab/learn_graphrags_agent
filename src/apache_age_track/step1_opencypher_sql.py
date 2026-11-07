"""
[Apache AGE Track Step 1] PostgreSQL 내 openCypher(cypher 함수) 기본 질의 실습
(src/03_apache_age_track/2_step1_opencypher_sql.py)

- 실습 내용:
  1. PostgreSQL 환경에서 `LOAD 'age';` 및 `SET search_path` 동작 이해
  2. `SELECT * FROM cypher('demon_slayer', $$ MATCH ... RETURN ... $$) as (...)` 구문 실행
  3. `agtype` 정점/간선 데이터 파싱 및 관계형 SQL 조인 결합 기초 실습
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.apache_age_track.common_age import get_age_connection, parse_agtype

def test_basic_opencypher():
    print("=" * 80)
    print("🚀 [Step 1 openCypher in SQL] PostgreSQL 내 그래프 쿼리 실습")
    print("=" * 80)

    conn = get_age_connection()
    try:
        with conn.cursor() as cur:
            # 1. 정점 개수 확인
            print("\n1️⃣ 정점(Node) 라벨별 통계 조회:")
            cur.execute("""
            SELECT * FROM cypher('demon_slayer', $$
                MATCH (n)
                RETURN labels(n) as label, count(n) as cnt
            $$) as (label agtype, cnt agtype);
            """)
            for row in cur.fetchall():
                print(f"  - 라벨: {parse_agtype(row[0])}, 개수: {parse_agtype(row[1])}")

            # 2. 탄지로의 주요 전투 관계 조회
            print("\n2️⃣ 카마도 탄지로의 전투(FIGHTS) 간선 조회:")
            cur.execute("""
            SELECT * FROM cypher('demon_slayer', $$
                MATCH (a:인간 {name: '카마도 탄지로'})-[r:FIGHTS]->(b)
                RETURN a.name, type(r), b.name, r.episode_number, r.outcome
            $$) as (hero agtype, rel agtype, enemy agtype, ep agtype, outcome agtype);
            """)
            records = cur.fetchall()
            for r in records[:5]:
                print(f"  ⚔️ [{parse_agtype(r[3])}] {parse_agtype(r[0])} --({parse_agtype(r[1])})--> {parse_agtype(r[2])} (결과: {parse_agtype(r[4])})")

            # 3. 관계형 테이블과 그래프 질의 결합 예제
            print("\n3️⃣ RDBMS 테이블과 그래프 쿼리 결합 예시:")
            cur.execute("""
            SELECT e.id, e.name, e.label
            FROM demon_slayer_node_embeddings e
            WHERE e.label = '도깨비';
            """)
            demons = cur.fetchall()
            print(f"  👹 PostgreSQL 관계형 테이블에 저장된 도깨비 목록 ({len(demons)}명):")
            for d in demons:
                print(f"     - [{d[0]}] {d[1]}")

        print("\n" + "=" * 80)
        print("✅ [Step 1] openCypher SQL 실행 실습 성공 완료")
        print("=" * 80)

    finally:
        conn.close()

if __name__ == "__main__":
    test_basic_opencypher()
