from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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
    """<think> 태그를 제거하는 커스텀 LLM"""
    def invoke(self, input: str) -> LLMResponse:
        response = super().invoke(input)
        content = response.content
        
        # 디버깅: 원본 응답 길이 로그
        print(f"  📊 Raw LLM response length: {len(content)} characters")
        if len(content) < 2000:
            print(f"  🔍 Raw content preview: {content[:1000]}...")

        # <think>...</think> 블록 제거
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.replace('</think>', '')
        
        # 1. 마크다운 코드 블록 제거 (```cypher ... ``` 또는 ``` ... ```)
        # 가장 확실한 방법: 코드 블록이 있으면 무조건 그것만 사용
        code_block_match = re.search(r'```(?:cypher)?(.*?)```', content, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            content = code_block_match.group(1).strip()
        else:
            # 코드 블록이 없으면, 혹시라도 MATCH...RETURN이 있는지 확인하지만
            # 이 모델은 너무 말이 많아서 위험함.
            # 아주 엄격하게 MATCH ( 로 시작하는 것만 허용
            match_query = re.search(r'(MATCH\s*\(.*RETURN[\s\S]+?(?:ORDER BY[\s\S]+?)?)(?:$|;)', content, re.DOTALL | re.IGNORECASE)
            if match_query:
                content = match_query.group(1).strip()
            else:
                print("  ⚠️ Warning: No code block or valid Cypher found.")
        
        # 공백 정리
        content = content.strip()
        
        # 만약 여전히 'MATCH'가 없다면 에러가 날 수 있음.
        # 하지만 빈 문자열을 보내면 SyntaxError가 나므로, 최소한의 방어 로직
        if not content:
             print("  ⚠️ Warning: Extracted content is empty.")

        response.content = content
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

# ============================================================================
# 1. 스키마 정의 (Schema Definition)
# ============================================================================

SCHEMA_NODES = """
## 노드 타입 (Node Types):
- 인간: {id: STRING, name: STRING, embedding: LIST[FLOAT]}
- 도깨비: {id: STRING, name: STRING, embedding: LIST[FLOAT]}
"""

SCHEMA_RELATIONSHIPS = """
## 관계 속성 (Relationship Properties): 
### 공통 속성 (모든 관계): - episode_number: STRING (예: "S01E01") - season: INTEGER - episode: INTEGER - context: STRING (전체 상황 설명) - description: STRING (구체적 행동 설명) - embedding: LIST[FLOAT] ### 전투 관련 속성 (FIGHTS, BATTLES, DEFEATS, ATTACKS): - outcome: STRING (승리/패배/무승부) - action: STRING (구체적 행동, 예: "대시와 antidote 사용") - technique: STRING (사용 기술, 예: "Blood Demon Art", "물의 호흡") - method: STRING (전투 방식) - effectiveness: STRING (효과성: high/medium/low) - enemy: STRING (적 이름) ### 보호/지원 관련 속성 (PROTECTS, SAVES, RESCUES, DEFENDS, HEALS): - role: STRING (역할, 예: "주요 가족 구성원", "멘토") - method: STRING (보호/치료 방법, 예: "전투 후 회복술 투여") - effectiveness: STRING (효과성: high/medium/low) - duration: STRING (지속 기간: 단기/장기) - to: STRING (보호/지원 대상) - subject: STRING (행위 주체) ### 이벤트 관련 속성: - event: STRING (특별 사건 설명) - commendation: STRING (인정/평가 내용)
"""

RELATIONSHIP_TYPES = {
    "전투": ["FIGHTS", "BATTLES", "DEFEATS", "ATTACKS"],
    "보호/지원": ["PROTECTS", "SAVES", "RESCUES", "DEFENDS", "SUPPORTS", "HEALS"],
    "관계": ["SIBLING_OF", "FAMILY_OF", "ALLY_OF", "ENEMY_OF"],
    "학습": ["TRAINS", "TRAINS_WITH", "TEACHES", "GUIDES"],
    "만남": ["MEETS", "ENCOUNTERS", "REUNITES_WITH", "JOINS"],
    "기타": ["TRANSFORMS"]
}

SCHEMA_PATTERNS = """
## 주요 관계 패턴 (Relationship Patterns):
(:인간)-[:FIGHTS|BATTLES|DEFEATS]->(:도깨비)
(:인간)-[:PROTECTS|SAVES|DEFENDS]->(:인간)
(:인간)-[:TRAINS|TRAINS_WITH]->(:인간)
(:인간)-[:SIBLING_OF|FAMILY_OF|ALLY_OF]->(:인간)
(:도깨비)-[:ATTACKS]->(:인간)
"""

MAIN_CHARACTERS = """
## 주요 캐릭터:
**인간**: 카마도 탄지로, 카마도 네즈코, 토미오카 기유, 우로코다키 사콘지, 
         아가츠마 젠이츠, 하시비라 이노스케, 렌고쿠 쿄쥬로, 코쵸우 시노부
**도깨비**: 키부츠지 무잔, 루이, 엔무, 쿄우가이
"""

# 전체 스키마 조합
FULL_SCHEMA = f"""
{SCHEMA_NODES}
{SCHEMA_RELATIONSHIPS}
{SCHEMA_PATTERNS}
{MAIN_CHARACTERS}
"""

# ============================================================================
# 2. 쿼리 예제 (Query Examples)
# ============================================================================

EXAMPLES_SINGLE_CHARACTER = [
    {
        "category": "단일 캐릭터 활약",
        "user_input": "카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?",
        "cypher": """
```cypher
MATCH (n:인간 {name: '카마도 탄지로'})-[r]-(m)
WHERE r.episode_number IS NOT NULL AND r.season = 1
RETURN n { .* , embedding: NULL } as n, 
       r { .* , embedding: NULL } as r, 
       m { .* , embedding: NULL } as m, 
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS relationship_type
ORDER BY r.episode
```
        """.strip()
    },
    {
        "category": "역할 분석",
        "user_input": "토미오카 기유는 어떤 역할을 했는지 에피소드별로 알려줘",
        "cypher": """
```cypher
MATCH (n:인간 {name: '토미오카 기유'})-[r]-(m)
WHERE r.episode_number IS NOT NULL
RETURN n { .* , embedding: NULL } as n, 
       r { .* , embedding: NULL } as r, 
       m { .* , embedding: NULL } as m,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS relationship_type
ORDER BY r.season, r.episode
```
        """.strip()
    }
]

EXAMPLES_RELATIONSHIPS = [
    {
        "category": "두 캐릭터 관계",
        "user_input": "카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어?",
        "cypher": """
```cypher
MATCH (a:인간 {name: '카마도 탄지로'})-[r]-(b:인간 {name: '카마도 네즈코'})
WHERE r.episode_number IS NOT NULL
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS relationship_type
ORDER BY r.season, r.episode
```
        """.strip()
    },
    {
        "category": "동료 관계",
        "user_input": "아가츠마 젠이츠와 하시비라 이노스케의 관계는?",
        "cypher": """
```cypher
MATCH (a:인간 {name: '아가츠마 젠이츠'})-[r]-(b:인간 {name: '하시비라 이노스케'})
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS relationship_type
ORDER BY r.season, r.episode
```
        """.strip()
    }
]

EXAMPLES_BATTLES = [
    {
        "category": "특정 전투",
        "user_input": "카마도 탄지로가 루이와 싸운 에피소드는?",
        "cypher": """
```cypher
MATCH (a:인간 {name: '카마도 탄지로'})-[r:FIGHTS|BATTLES]-(b:도깨비 {name: '루이'})
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS relationship_type,
       r.outcome AS result
ORDER BY r.season, r.episode
```
        """.strip()
    },
    {
        "category": "모든 전투",
        "user_input": "카마도 탄지로의 모든 전투 기록을 보여줘",
        "cypher": """
```cypher
MATCH (a:인간 {name: '카마도 탄지로'})-[r:FIGHTS|BATTLES|DEFEATS]-(b:도깨비)
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS relationship_type,
       r.outcome AS result
ORDER BY r.season, r.episode
```
        """.strip()
    }
]

# 모든 예제 통합
ALL_EXAMPLES = EXAMPLES_SINGLE_CHARACTER + EXAMPLES_RELATIONSHIPS + EXAMPLES_BATTLES

# ============================================================================
# 3. Cypher 작성 규칙 (Cypher Syntax Rules)
# ============================================================================

CYPHER_RULES = """
## Cypher 작성 규칙:

1. **속성 접근**: 단일 중괄호 사용 {name: 'value'}
2. **레이블 지정**: :인간, :도깨비
3. **이름 매칭**: {name: '캐릭터명'} 형식
4. **관계 타입**: 영문 대문자 (FIGHTS, PROTECTS 등)
5. **정렬**: 항상 season, episode 순서로 ORDER BY
6. **필수 필터**: episode_number IS NOT NULL
7. **임베딩 제외 (필수)**: RETURN 시 `{{ .* , embedding: NULL }}` 구문을 사용하여 노드와 관계의 embedding 속성을 반드시 제외해야 합니다.

## 자주 쓰는 패턴:
- 단일 캐릭터: MATCH (n:인간 {name: '이름'})-[r]-(m)
- 두 캐릭터: MATCH (a {name: '이름1'})-[r]-(b {name: '이름2'})
- 전투만: -[r:FIGHTS|BATTLES|DEFEATS]-
- 시즌 필터: WHERE r.season = 1
"""

# ============================================================================
# 4. 최종 프롬프트 템플릿 (Final Prompt Template)
# ============================================================================

SYSTEM_PROMPT = f"""You are a Neo4j Cypher Query Generator.
Task: Convert User Question to Cypher Query.

## Schema
{{FULL_SCHEMA}}

## Rules
{{CYPHER_RULES}}
"""

MAIN_PROMPT_TEMPLATE = """
## Examples
{examples}

## User Question
{query_text}

## Instructions
1. Output the Cypher query inside a markdown code block: ```cypher ... ```
2. Use valid Cypher syntax only.
3. Exclude embedding properties in RETURN.
4. Use `relationship_type` as alias for `type(r)`.

OUTPUT:
"""

# ============================================================================
# 5. Retriever 설정
# ============================================================================

# Escape braces in the schema and rules because they will be formatted
escaped_system_prompt = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    examples=[
        f"USER INPUT: '{ex['user_input']}'\nQUERY: {ex['cypher']}" 
        for ex in ALL_EXAMPLES
    ],
    custom_prompt=escaped_system_prompt + "\n\n" + MAIN_PROMPT_TEMPLATE
)

def llm_call(prompt: str) -> str:
    # Use the cleaning LLM instance
    response = llm.invoke(prompt)
    return response.content

llm_cal = llm_call  # backwards compatibility alias

ANSWER_GENERATION_PROMPT = """당신은 애니메이션 "귀멸의 칼날"의 전문가입니다.
아래 데이터베이스 검색 결과를 바탕으로 사용자의 질문에 대해 **최대한 상세하고 풍부하게** 답변하세요.

## 사용자 질문:
{question}

## 데이터베이스 검색 결과:
{context}

## 답변 작성 규칙:

### 1. 상세한 에피소드별 서술 (중요)
- 각 에피소드에서 발생한 사건을 **육하원칙**에 따라 구체적으로 서술하세요.
- 단순히 "싸웠다"가 아니라, "**어떤 기술**을 사용하여 **어떻게** 싸웠는지, 결과는 어떠했는지" 묘사하세요.
- **감정선**과 **대사**의 뉘앙스를 포함하여 스토리의 몰입감을 높이세요.
- 검색 결과에 있는 모든 관련 에피소드를 빠짐없이 포함하세요.

### 2. 자연스러운 관계 표현
- DB의 관계명을 그대로 쓰지 말고 자연스러운 문장으로 변환하세요.
  - FIGHTS/BATTLES → "치열한 전투를 벌이다", "격돌하다"
  - PROTECTS → "몸을 던져 지키다", "필사적으로 보호하다"
  - TRAINS → "혹독한 훈련을 지도하다", "가르침을 받다"
  - DEFEATS → "쓰러뜨리다", "목을 베다"

### 3. 답변 구조
```
[서론: 질문에 대한 전체적인 요약 1-2문장]

## 에피소드별 상세 기록

### 📺 시즌 1 에피소드 [번호]
- **주요 사건**: [핵심 사건 명시]
- **상세 내용**: 
  [검색 결과를 바탕으로 한 상세한 줄거리 서술. 
   누가, 어디서, 무엇을, 어떻게 했는지 구체적으로 작성.]

... (모든 에피소드 반복) ...

[결론: 캐릭터의 성장이나 관계의 변화에 대한 통찰]
```

### 4. 주의사항
- **절대 요약하지 마세요.** 사용자에게 정보를 충분히 제공하는 것이 목표입니다.
- 검색 결과에 없는 내용은 꾸며내지 마세요.
- "데이터베이스에서", "검색 결과에 따르면" 같은 표현은 쓰지 마세요.

## 답변:"""

def graphrag_pipeline(user_question):

    # 1 질문 -> cypher query -> 결과 리스트 반환
    try:
        result = retriever.search(query_text=user_question)
    except Exception as e:
        traceback.print_exc()
        return f"검색 중 오류가 발생했습니다: {e}"

    # 2 Cypher Query 확인
    cypher_used = None
    if hasattr(result, "metadata") and isinstance(result.metadata, dict):
        cypher_used = result.metadata.get("cypher")
    elif isinstance(result, dict):
        cypher_used = result.get("metadata", {}).get("cypher") or result.get("cypher")
    print("생성된 Cypher Query:")
    print(cypher_used)

    # 3 결과 확인
    result_items = getattr(result, "items", []) if hasattr(result, "items") else (result.get("items", []) if isinstance(result, dict) else [])
    print("지식그래프에 찾은 결과")
    print(result_items)

    if not result_items:
        return "데이터베이스에서 관련 정보를 찾을 수 없습니다."

    # 4 결과 기반으로 프롬프트 완성
    context_list = []
    for item in result_items:
        # 임베딩은 이미 Cypher에서 제외되었으므로 바로 사용
        context_list.append(str(item.content))

    full_context = "\n".join(context_list)

    # ANSWER_GENERATION_PROMPT 사용
    full_prompt = ANSWER_GENERATION_PROMPT.format(
        question=user_question,
        context=full_context
    )

    print("완성 프롬프트")
    print(f"Prompt length: {len(full_prompt)}")
    print("="*50)
    # 3 완성된 프롬프트로 최종 답변 생성
    final_result = llm_cal(full_prompt)
    return final_result

if __name__=="__main__":
    queries = [
        # "카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어? 에피소드별로 정리해줘.",
        # "토미오카 기유는 시즌 1에서 어떤 역할을 했는지 에피소드별로 알려줘.",
        # "카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?",
        "아가츠마 젠이츠와 하시비라 이노스케는 언제 처음 만났어?",
    ]
    
    for query in queries:
        print(query)
        print("-"*100)
        print(graphrag_pipeline(query))
        print("-"*100)


