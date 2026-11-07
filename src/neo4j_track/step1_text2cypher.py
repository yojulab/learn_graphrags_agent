"""
[Neo4j Track Step 1] 기초 순수 Text2Cypher GraphRAG 에이전트 실습
(src/neo4j_track/step1_text2cypher.py)

- 실습 내용:
  1. LPG 스키마 프롬프트 제공
  2. LLM(Exaone-4.0)을 통한 사용자 자연어 질문 ➔ Cypher MATCH 쿼리 생성
  3. CleanOpenAILLM 태그 정제
  4. Cypher 질의 결과 컨텍스트 기반 답변 합성
"""

import sys
import re
import traceback
from pathlib import Path
from typing import Dict, Any, List
from neo4j_graphrag.retrievers import Text2CypherRetriever

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.neo4j_track.common_neo4j import get_neo4j_driver, CleanOpenAILLM, SCHEMA_NODES, SCHEMA_RELATIONSHIPS

driver = get_neo4j_driver()
llm = CleanOpenAILLM(
    model_name=config.LLM_MODEL,
    model_params={"max_tokens": 2000, "temperature": 0},
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

EXAMPLES = [
    {
        "user_input": "카마도 탄지로는 시즌 1에서 어떤 활약을 했어?",
        "cypher": "MATCH (n:인간 {name: '카마도 탄지로'})-[r]-(m) WHERE r.episode_number IS NOT NULL RETURN n.name as hero, type(r) as rel_type, m.name as target, r.episode_number as ep, r.context as context"
    },
    {
        "user_input": "카마도 탄지로와 루이가 싸운 에피소드는?",
        "cypher": "MATCH (a:인간 {name: '카마도 탄지로'})-[r:FIGHTS|BATTLES]-(b:도깨비 {name: '루이'}) RETURN a.name, type(r) as rel_type, b.name, r.episode_number as ep, r.context as context, r.outcome as outcome"
    }
]

# 포맷팅 중괄호 충돌 방지를 위해 스키마 내 중괄호 이스케이프
safe_schema_nodes = SCHEMA_NODES.replace("{", "{{").replace("}", "}}")
safe_schema_rels = SCHEMA_RELATIONSHIPS.replace("{", "{{").replace("}", "}}")

CUSTOM_PROMPT = f"""You are a Neo4j Cypher query generator for a Demon Slayer knowledge graph.
Convert the user question into a valid Cypher query using the schema below.

## Schema
{safe_schema_nodes}
{safe_schema_rels}

## Instructions
1. Write ONLY the Cypher query inside a markdown code block: ```cypher ... ```
2. Use ONLY MATCH queries (no CREATE, DELETE, SET).
3. Do not include embedding properties in RETURN.

## Examples
{{examples}}

## User Question:
{{query_text}}

Cypher Query:"""

retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=f"{SCHEMA_NODES}\n{SCHEMA_RELATIONSHIPS}",
    examples=[f"User: {e['user_input']}\nCypher: ```cypher\n{e['cypher']}\n```" for e in EXAMPLES],
    custom_prompt=CUSTOM_PROMPT
)

def run_text2cypher(question: str) -> str:
    print(f"\n💬 [Step 1 Text2Cypher] 사용자 질문: {question}")
    try:
        res = retriever.search(query_text=question)
        items = getattr(res, "items", []) or []
        cypher = res.metadata.get("cypher") if hasattr(res, "metadata") and isinstance(res.metadata, dict) else "N/A"
        print(f"  📝 생성된 Cypher: {cypher}")
        print(f"  📊 조회된 레코드: {len(items)}건")

        if not items:
            return "지식 그래프 데이터상 확인되지 않는 정보입니다."

        context = "\n".join([str(item.content) for item in items[:15]])
        prompt = f"""당신은 귀멸의 칼날 지식그래프 질의응답 전문가입니다.
다음 지식그래프 검색 결과를 바탕으로 질문에 정확하고 구체적으로 한국어로 답변하세요.

## 검색 결과 컨텍스트:
{context}

## 사용자 질문:
{question}

## 답변:"""
        ans = llm.invoke(prompt)
        return ans.content
    except Exception as e:
        traceback.print_exc()
        return f"쿼리 실행 중 오류가 발생했습니다: {e}"

if __name__ == "__main__":
    test_q = "카마도 탄지로가 루이와 싸운 에피소드는 몇 화이고 어떤 기술을 썼어?"
    print(run_text2cypher(test_q))
