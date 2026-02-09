from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv
import config
import openai
import os
import re

import traceback

## OpenAI 클라이언트 선언
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

## Neo4j 드라이버와 리트리버 선언
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.llm.types import LLMResponse

class CleanOpenAILLM(OpenAILLM):
    def invoke(self, input: str) -> LLMResponse:
        response = super().invoke(input)
        content = response.content
        # Remove <think>...</think> blocks including the tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Remove any remaining closing tags and preceding content (handling malformed/partial output)
        content = re.sub(r'.*?</think>', '', content, flags=re.DOTALL)
        response.content = content.strip()
        return response

## Neo4j 드라이버와 리트리버 선언
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
llm = CleanOpenAILLM(
    model_name=config.LLM_MODEL,
    model_params={
        "max_tokens": 2000,
        "temperature": 0,
    },
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

examples = [
    "USER INPUT: '토미오카 기유는 시즌 1에서 어떤 역할을 했는지 에피소드별로 알려줘.' QUERY: MATCH (n {{name: '토미오카 기유'}})-[r]-(m) RETURN n, r, m, properties(r) AS rel_props ORDER BY r.episode_number",
    "USER INPUT: '카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?' QUERY: MATCH (n {{name: '카마도 탄지로'}})-[r]-(m) RETURN n, r, m, properties(r) AS rel_props ORDER BY r.episode_number",
    "USER INPUT: '카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어? 에피소드별로 정리해줘.' QUERY: MATCH (n {{name: '카마도 탄지로'}})-[r]-(m {{name: '카마도 네즈코'}}) RETURN n, r, m, properties(r) AS rel_props ORDER BY r.episode_number"
]




# Define the schema manually based on the known data to help the LLM
known_schema = """
Node Labels: [인간, 도깨비]
Relationship Types: [FIGHTS, PROTECTS, TRAINS, KNOWS, FAMILY_OF, ALLY_OF, ENEMY_OF, DEFEATS, SAVES, MEETS]
Node Properties: id, name
Relationship Properties: episode_number, outcome
"""

# Note: Double braces {{}} are needed to escape Python's .format() parsing
# The actual Cypher output should use single braces {}
custom_prompt = """Task: Generate a Cypher statement to query a Neo4j graph database.

Schema:
""" + known_schema + """

Cypher Syntax Rules:
- Properties use single curly braces in Cypher: {{name: 'value'}}
- Use node labels like :인간 or :도깨비
- Put character names in the 'name' property, e.g.: (n:인간 {{name: '카마도 탄지로'}})

Examples:
{examples}

User Question:
{query_text}

OUTPUT ONLY THE CYPHER QUERY. NO EXPLANATION.
"""

retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,  
    examples=examples,
    custom_prompt=custom_prompt
)

def llm_cal(prompt: str) -> str:
    # Use the cleaning LLM instance
    response = llm.invoke(prompt)
    return response.content

def graphrag_pipeline(user_question):

    # 1 질문 -> cypher query -> 결과 리스트 반환
    try:
        result = retriever.search(query_text=user_question)
    except Exception as e:
        traceback.print_exc()
        return f"검색 중 오류가 발생했습니다: {e}"

    # 2 Cypher Query 확인
    cypher_used = result.metadata.get("cypher")
    print("생성된 Cypher Query:")
    print(cypher_used)


    # 3 결과 확인
    result_items = result.items
    print("지식그래프에 찾은 결과")
    print(result_items)

    if not result_items:
        return "데이터베이스에서 관련 정보를 찾을 수 없습니다."

    # 4 결과 기반으로 프롬프트 완성
    context_list = []
    for item in result_items:
        raw = str(item.content)
        # element_id=... 부분만 제거
        cleaned = re.sub(r"element_id='[^']*'\s*", "", raw)
        context_list.append(cleaned)

    full_context = "\n".join(context_list)

    full_prompt = f"""
    아래의 데이터베이스 결과만을 참고하여 사용자의 질문에 답변해주세요.  
    데이터베이스 결과를 그대로 노출하지 말고, 자연스러운 서술로 정리해 주세요.  
    사용자의 질문: {user_question}  
    데이터베이스 결과: {full_context}  

    ### 조건
    - 그래프DB의 관계명(예: DEFENDS, SIBLING_OF, REUNITES_WITH 등)은 그대로 쓰지 말고,
    맥락에 맞는 자연스러운 한국어 문장으로 풀어 설명하세요.
    - 에피소드별로 일어난 사건은 간결하고 이해하기 쉽게 정리하세요.
    - 응답은 마치 스토리를 설명하듯 자연스럽게 작성하세요.
    """
    print("완성 프롬프트")
    print(full_prompt)
    # 3 완성된 프롬프트로 최종 답변 생성
    final_result = llm_cal(full_prompt)
    return final_result

if __name__=="__main__":
    queries = [

    # "카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?",
    # "토미오카 기유는 시즌 1에서 어떤 역할을 했는지 에피소드별로 알려줘.",
    "카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어? 에피소드별로 정리해줘.",
        ]
    
    for query in queries:
        print(query)
        print("-"*100)
        print(graphrag_pipeline(query))
        print("-"*100)


