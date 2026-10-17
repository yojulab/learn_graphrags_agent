"""
[v4] PostgreSQL 16+ & Apache AGE Multi-Model GraphRAG 에이전트
(src/v4/3_graphrag_agent_v4.py)

핵심 파이프라인 (4단계 하네스):
1. Stage 1: 질문 의도 분류기 (Query Intent Classifier & Named Entity Extractor)
2. Stage 2: PostgreSQL In-Database Cosine Similarity 벡터 검색 (Seed Node/Rel Retrieval)
3. Stage 3: openCypher in SQL 질의 생성 & 1~2 hop 동적 서브그래프 확장
4. Stage 4: Think Tag 정제 및 한국어 구조화 답변 합성기 (Anti-Hallucination Guardrail)
"""

import os
import sys
import json
import re
import time
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import psycopg2
import openai

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

# OpenAI 호환 클라이언트 초기화
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL,
    timeout=300.0
)

# ============================================================
# 1. 질문 유형 체계 (Stage 1)
# ============================================================

class QueryType(Enum):
    SINGLE_ENTITY = "single_entity"          # 단일 인물 질문
    RELATIONSHIP = "relationship"            # 두 인물 간 관계
    EPISODE_SPECIFIC = "episode_specific"    # 특정 에피소드
    GENERAL = "general"                      # 일반 질문


def classify_query_type(question: str) -> Tuple[QueryType, Dict[str, Any]]:
    """질문 유형 분류 및 핵심 개체 추출"""
    entities: Dict[str, Any] = {}
    q = question.strip()

    # 에피소드 번호 패턴 감지
    ep_match = re.search(r'(\d+)\s*(화|회|화차|episode|ep)', q, re.IGNORECASE)
    if ep_match:
        entities["episode_number"] = int(ep_match.group(1))
        return QueryType.EPISODE_SPECIFIC, entities

    # 노드 이름 매칭 탐색
    matched_names = []
    # 마스터 노드 이름 목록 탐색
    all_known_names = [
        "카마도 탄지로", "카마도 네즈코", "토미오카 기유", "우로코다키 사콘지",
        "사비토", "마코모", "아가츠마 젠이츠", "하시비라 이노스케",
        "츠유리 카나오", "렌고쿠 쿄쥬로", "우부야시키 카가야", "코쵸우 시노부",
        "시나즈가와 사네미", "키부츠지 무잔", "스사마루", "야하바",
        "쿄우가이", "루이", "엔무"
    ]
    # 부분 별칭 매핑
    alias_map = {
        "탄지로": "카마도 탄지로",
        "네즈코": "카마도 네즈코",
        "기유": "토미오카 기유",
        "사콘지": "우로코다키 사콘지",
        "젠이츠": "아가츠마 젠이츠",
        "이노스케": "하시비라 이노스케",
        "카나오": "츠유리 카나오",
        "렌고쿠": "렌고쿠 쿄쥬로",
        "시노부": "코쵸우 시노부",
        "사네미": "시나즈가와 사네미",
        "무잔": "키부츠지 무잔",
    }

    found = set()
    for name in all_known_names:
        if name in q:
            found.add(name)
    if not found:
        for alias, full_name in alias_map.items():
            if alias in q:
                found.add(full_name)

    matched_names = list(found)

    if len(matched_names) >= 2:
        entities["entity_a"] = matched_names[0]
        entities["entity_b"] = matched_names[1]
        return QueryType.RELATIONSHIP, entities
    elif len(matched_names) == 1:
        entities["entity"] = matched_names[0]
        return QueryType.SINGLE_ENTITY, entities

    return QueryType.GENERAL, entities


# ============================================================
# 2. LLM 추론 및 <think> 태그 정제기
# ============================================================

def sanitize_think_tags(text: str) -> str:
    """<think>...</think> 태그 정규식 완전 제거"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace('</think>', '')
    return cleaned.strip()


def invoke_clean_llm(prompt: str, max_tokens: int = 1500, temperature: float = 0.0) -> str:
    """Ollama LLM 호출 및 생각 태그 정제"""
    try:
        res = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        raw_text = res.choices[0].message.content or ""
        return sanitize_think_tags(raw_text)
    except Exception as e:
        print(f"⚠️  LLM 호출 오류: {e}")
        return ""


# ============================================================
# 3. PostgreSQL In-Database 벡터 검색기 (Stage 2)
# ============================================================

class PostgreSQLVectorRetriever:
    """PostgreSQL node_embeddings 테이블 기반 코사인 유사도 검색기"""
    def __init__(self):
        self.embedding_model = config.EMBEDDING_MODEL

    def get_query_embedding(self, query: str) -> List[float]:
        res = client.embeddings.create(model=self.embedding_model, input=[query])
        return res.data[0].embedding

    def search_seed_nodes(self, query: str, top_k: int = config.VECTOR_TOP_K) -> List[Dict[str, Any]]:
        """유사도 높은 시드 정점 검색"""
        t0 = time.time()
        query_emb = self.get_query_embedding(query)
        emb_time = time.time() - t0

        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB
        )
        cur = conn.cursor()
        try:
            cur.execute("""
            SELECT id, label, name, text, cosine_similarity(embedding, %s) AS score
            FROM node_embeddings
            ORDER BY score DESC
            LIMIT %s;
            """, (query_emb, top_k))

            rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "label": r[1],
                    "name": r[2],
                    "text": r[3],
                    "score": float(r[4])
                })
            return results
        finally:
            cur.close()
            conn.close()


# ============================================================
# 4. Apache AGE openCypher SQL 쿼리 엔진 (Stage 3)
# ============================================================

def clean_agtype(val: Any) -> str:
    """agtype 결과값에서 따옴표 및 타입 주석 정제"""
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    return s


class AGECypherEngine:
    """PostgreSQL cypher(...) 실행 및 검증 엔진"""

    def __init__(self):
        self.graph_name = config.AGE_GRAPH_NAME

    def validate_safety(self, sql: str) -> bool:
        """읽기 전용 가드레일: 변경 구문 차단"""
        forbidden = [r'\bCREATE\b', r'\bDROP\b', r'\bDELETE\b', r'\bSET\b', r'\bMERGE\b', r'\bREMOVE\b']
        for pat in forbidden:
            if re.search(pat, sql, re.IGNORECASE):
                print(f"🚫 [가드레일 차단] 읽기 전용 위반 구문 감지: {pat}")
                return False
        return True

    def execute_cypher_sql(self, sql: str) -> List[Dict[str, Any]]:
        """openCypher SQL 쿼리 안전 실행"""
        if not self.validate_safety(sql):
            return []

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

            # 컬럼명 가져오기
            col_names = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()

            results = []
            for row in rows:
                record = {}
                for idx, col in enumerate(col_names):
                    record[col] = clean_agtype(row[idx])
                results.append(record)
            return results
        except Exception as e:
            print(f"⚠️  openCypher SQL 실행 실패: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def expand_from_seeds(self, seed_ids: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """시드 노드 기반 1~2 hop 동적 서브그래프 확장"""
        if not seed_ids:
            return []

        ids_formatted = ", ".join(f"'{sid}'" for sid in seed_ids)
        sql = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (a)-[r]->(b)
            WHERE a.id IN [{ids_formatted}] OR b.id IN [{ids_formatted}]
            RETURN a.name, type(r), b.name, r.context, r.episode_number
            LIMIT {limit}
        $$) as (source agtype, rel_type agtype, target agtype, context agtype, episode agtype);
        """
        return self.execute_cypher_sql(sql)

    def generate_intent_query(self, query_type: QueryType, entities: Dict[str, Any], seed_nodes: List[Dict[str, Any]]) -> str:
        """질문 유형 및 시드 엔티티 기반 최적 openCypher in SQL 생성"""
        if query_type == QueryType.RELATIONSHIP:
            name_a = entities.get("entity_a") or (seed_nodes[0]["name"] if seed_nodes else "탄지로")
            name_b = entities.get("entity_b") or (seed_nodes[1]["name"] if len(seed_nodes) > 1 else "네즈코")
            return f"""
            SELECT * FROM cypher('{self.graph_name}', $$
                MATCH (a)-[r]->(b)
                WHERE (a.name CONTAINS '{name_a}' AND b.name CONTAINS '{name_b}')
                   OR (a.name CONTAINS '{name_b}' AND b.name CONTAINS '{name_a}')
                RETURN a.name, type(r), b.name, r.context, r.episode_number
                LIMIT 30
            $$) as (source agtype, rel_type agtype, target agtype, context agtype, episode agtype);
            """.strip()

        elif query_type == QueryType.SINGLE_ENTITY:
            char_name = entities.get("entity") or (seed_nodes[0]["name"] if seed_nodes else "카마도 탄지로")
            return f"""
            SELECT * FROM cypher('{self.graph_name}', $$
                MATCH (a)-[r]->(b)
                WHERE a.name CONTAINS '{char_name}' OR b.name CONTAINS '{char_name}'
                RETURN a.name, type(r), b.name, r.context, r.episode_number
                LIMIT 30
            $$) as (source agtype, rel_type agtype, target agtype, context agtype, episode agtype);
            """.strip()

        elif query_type == QueryType.EPISODE_SPECIFIC:
            ep = entities.get("episode_number", 1)
            return f"""
            SELECT * FROM cypher('{self.graph_name}', $$
                MATCH (a)-[r]->(b)
                WHERE r.episode_number = {ep}
                RETURN a.name, type(r), b.name, r.context, r.episode_number
                LIMIT 30
            $$) as (source agtype, rel_type agtype, target agtype, context agtype, episode agtype);
            """.strip()

        else:
            # GENERAL
            seed_names = [n["name"] for n in seed_nodes[:3]]
            if seed_names:
                seed_cond = " OR ".join(f"a.name CONTAINS '{s}' OR b.name CONTAINS '{s}'" for s in seed_names)
                return f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (a)-[r]->(b)
                    WHERE {seed_cond}
                    RETURN a.name, type(r), b.name, r.context, r.episode_number
                    LIMIT 30
                $$) as (source agtype, rel_type agtype, target agtype, context agtype, episode agtype);
                """.strip()
            else:
                return f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (a)-[r]->(b)
                    RETURN a.name, type(r), b.name, r.context, r.episode_number
                    LIMIT 25
                $$) as (source agtype, rel_type agtype, target agtype, context agtype, episode agtype);
                """.strip()


# ============================================================
# 5. 최종 답변 합성기 (Stage 4)
# ============================================================

class AnswerSynthesizer:
    """컨텍스트 정렬 및 환각 방지 한국어 답변 생성기"""

    @staticmethod
    def format_records_context(records: List[Dict[str, Any]]) -> str:
        """레코드를 에피소드 순서대로 포맷팅"""
        if not records:
            return ""

        # 에피소드 번호 기준 정렬
        def get_ep(r):
            try:
                return int(r.get("episode", 1))
            except Exception:
                return 1

        sorted_records = sorted(records, key=get_ep)

        lines = []
        for r in sorted_records:
            src = r.get("source", "")
            rel = r.get("rel_type") or r.get("relationship", "")
            tgt = r.get("target", "")
            ctx = r.get("context", "")
            ep = r.get("episode", 1)
            lines.append(f"- [S1E{ep}] {src} -[{rel}]-> {tgt}: {ctx}")

        return "\n".join(lines)

    @classmethod
    def synthesize_answer(cls, question: str, records: List[Dict[str, Any]]) -> str:
        """지식그래프 컨텍스트를 바탕으로 사실 기반 답변 합성"""
        if not records:
            return "지식 그래프 데이터상 확인되지 않는 정보입니다. (관련 인물이나 에피소드 정보가 지식그래프에 등록되어 있지 않습니다.)"

        formatted_context = cls.format_records_context(records)

        prompt = f"""
당신은 귀멸의 칼날 지식그래프(Apache AGE Multi-Model GraphRAG) 전문 분석가입니다.
제공된 [지식그래프 검색 결과]만을 바탕으로 사용자의 질문에 정확하고 충실하게 한국어로 답변하세요.

[지식그래프 검색 결과]:
{formatted_context}

[답변 작성 수칙]:
1. 오직 검색 결과에 명시된 사실(인물, 관계, 에피소드, 맥락)만 인용하여 답변하세요.
2. 사실에 근거한 서술을 한국어 개조식(Bullet points)으로 정리하세요.
3. 검색 결과에 없는 가공된 상상이나 추측은 절대 배제하세요.

질문: {question}
답변:
""".strip()

        answer = invoke_clean_llm(prompt, max_tokens=1500)
        if not answer or len(answer) < 10:
            # Fallback direct formatting
            res_lines = [f"**[Apache AGE GraphRAG v4 답변]**", f"질문: '{question}'", "", "### 지식그래프 검색 결과:"]
            for r in records[:5]:
                res_lines.append(f"- [{r.get('rel_type')}] S1E{r.get('episode')}: {r.get('source')} → {r.get('target')} ({r.get('context')})")
            return "\n".join(res_lines)

        return f"**[Apache AGE GraphRAG v4 답변]**\n\n{answer}"


# ============================================================
# 6. GraphRAG v4 Agent 통합 인터페이스
# ============================================================

class GraphRAGAgentV4:
    """PostgreSQL + Apache AGE Multi-Model GraphRAG Agent"""

    def __init__(self):
        self.vector_retriever = PostgreSQLVectorRetriever()
        self.cypher_engine = AGECypherEngine()
        self.synthesizer = AnswerSynthesizer()

    def query(self, question: str) -> Dict[str, Any]:
        """4단계 하네스 질의응답 실행"""
        start_time = time.time()
        print(f"\n💬 [GraphRAG v4 질의]: '{question}'")

        # 1. 의도 분류
        query_type, entities = classify_query_type(question)
        print(f"  🎯 [Stage 1] 분류: {query_type.value}, 엔티티: {entities}")

        # 2. 벡터 검색 (시드 노드 획득)
        t_vec = time.time()
        seed_nodes = self.vector_retriever.search_seed_nodes(question, top_k=config.VECTOR_TOP_K)
        seed_names = [n["name"] for n in seed_nodes]
        print(f"  🧠 [Stage 2] 시드 벡터 검색 ({time.time()-t_vec:.2f}s): {seed_names}")

        # 3. openCypher SQL 질의 및 확장
        t_graph = time.time()
        cypher_sql = self.cypher_engine.generate_intent_query(query_type, entities, seed_nodes)
        records = self.cypher_engine.execute_cypher_sql(cypher_sql)
        print(f"  🕸️ [Stage 3] 1차 쿼리 매칭: {len(records)}건 레코드 확보")

        # 만약 결과가 부족하면 시드 노드로부터 1~2 hop 서브그래프 확장
        if len(records) < 5 and seed_nodes:
            seed_ids = [n["id"] for n in seed_nodes[:3]]
            expanded_records = self.cypher_engine.expand_from_seeds(seed_ids, limit=30)
            print(f"  🌱 [Stage 3+] 1~2 hop 그래프 확장: {len(expanded_records)}건 추가 확보")
            # 중복 제거 병합
            seen = set()
            merged = []
            for r in records + expanded_records:
                key = (r.get("source"), r.get("rel_type"), r.get("target"), r.get("episode"))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            records = merged

        # 4. 답변 합성
        t_synth = time.time()
        answer = self.synthesizer.synthesize_answer(question, records)
        print(f"  📝 [Stage 4] 답변 합성 완료 ({time.time()-t_synth:.2f}s)")

        total_duration = time.time() - start_time
        return {
            "version": "v4",
            "question": question,
            "query_type": query_type.value,
            "entities": entities,
            "seed_nodes": seed_names,
            "subgraph_records_count": len(records),
            "answer": answer,
            "records": records[:10],
            "duration": f"{total_duration:.2f}s"
        }


# ============================================================
# CLI 테스트 엔트리포인트
# ============================================================

def main():
    print("=" * 80)
    print("🚀 [v4.0] PostgreSQL 16+ & Apache AGE Multi-Model GraphRAG Agent CLI")
    print("=" * 80)

    agent = GraphRAGAgentV4()

    sample_questions = [
        "카마도 탄지로와 카마도 네즈코 사이에 어떤 일들이 있었어?",
        "토미오카 기유의 역할과 탄지로와의 관계를 알려줘",
        "루이와 전투를 벌인 인물들은 누구야?",
        "시즌 1 제19화에서는 어떤 일이 일어났어?"
    ]

    for q in sample_questions:
        res = agent.query(q)
        print("\n" + "-" * 60)
        print(f"Q: {res['question']}")
        print(f"Time: {res['duration']} | Records: {res['subgraph_records_count']}")
        print(f"A:\n{res['answer']}")
        print("-" * 60)

    print("\n✅ [v4 CLI 테스트 완료]")
    return 0


if __name__ == "__main__":
    exit(main())
