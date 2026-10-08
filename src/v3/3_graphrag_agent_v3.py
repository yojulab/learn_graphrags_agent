"""
개선된 GraphRAG 에이전트 v3.0
- 하이브리드 검색: 벡터 검색 + Cypher 그래프 순회 결합
- 다단계 쿼리 검증 및 재시도
- 질문 유형 분류 및 템플릿 기반 쿼리 생성
- 컨텍스트 확장 전략
- 개선된 프롬프트 템플릿
- 확장된 Cypher 예제 (집계, 다중 홉, OPTIONAL MATCH, WITH 절)
- Neo4j 벡터 인덱스 활용 (entity_embeddings, relationship_embeddings)
"""

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import RetrieverResultItem
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
import openai
import os
import re
import time
import traceback
from enum import Enum
from tqdm import tqdm

# ============================================================
# OpenAI 클라이언트 초기화
# ============================================================
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

# ============================================================
# 질문 유형 정의
# ============================================================
class QueryType(Enum):
    SINGLE_ENTITY = "single_entity"          # 단일 인물 질문
    RELATIONSHIP = "relationship"            # 두 인물 간 관계
    EPISODE_SPECIFIC = "episode_specific"    # 특정 에피소드
    GENERAL = "general"                      # 일반 질문

# ============================================================
# 정리된 OpenAI LLM (think 태그 제거)
# ============================================================
class CleanOpenAILLM(OpenAILLM):
    """<think> 태그를 제거하는 커스텀 LLM"""
    def invoke(self, input: str) -> LLMResponse:
        response = super().invoke(input)
        content = response.content
        
        # 디버깅: 원본 응답 길이 로그
        print(f"  📊 Raw LLM response length: {len(content)} characters")
        if len(content) < 500:
            print(f"  🔍 Raw content preview: {content[:200]}...")

        # <think>...</think> 블록 제거 (비탐욕적 매칭)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # 닫히지 않은 <think> 태그 처리 (문자열 끝까지 제거)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        
        # 남은 종료 태그 제거
        content = content.replace('</think>', '')
        
        response.content = content.strip()
        return response

# ============================================================
# Neo4j 드라이버 및 LLM 초기화
# ============================================================
driver = GraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

llm = CleanOpenAILLM(
    model_name=config.LLM_MODEL,
    model_params={
        "max_tokens": 2000,
        "temperature": 0,  # 일관성을 위해 0으로 설정
    },
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL,
    timeout=300.0  # 5분 타임아웃 설정
)

# Embedder initialization for hybrid retrieval
from openai import OpenAI as OpenAIClient

embedder_client = OpenAIClient(
    api_key=config.OPENAI_API_KEY, 
    base_url=config.MODEL_API_URL,
    timeout=300.0  # 5분 타임아웃 설정
)

class OpenAIEmbedder:
    """Simple embedder for hybrid retrieval"""
    def __init__(self, client, model):
        self.client = client
        self.model = model
    
    def embed_query(self, text: str):
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding

embedder = OpenAIEmbedder(embedder_client, config.EMBEDDING_MODEL)


# ============================================================
# 스키마 정의
# ============================================================
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


# ============================================================
# 쿼리 템플릿 정의 (개선)
# ============================================================
QUERY_TEMPLATES = {
    "single_entity_timeline": """
MATCH (entity {{name: '{entity_name}'}})-[r]-(other)
WHERE r.episode_number IS NOT NULL
RETURN entity {{ .* , embedding: NULL }} as entity, 
       r {{ .* , embedding: NULL }} as r, 
       other {{ .* , embedding: NULL }} as other, 
       r.episode_number as episode, 
       r.context as description,
       type(r) as relationship_type,
       labels(other) as other_labels
ORDER BY r.season, r.episode
""",

    "relationship_between_two": """
MATCH (a {{name: '{entity_a}'}})-[r]-(b {{name: '{entity_b}'}})
WHERE r.episode_number IS NOT NULL
RETURN a {{ .* , embedding: NULL }} as a, 
       r {{ .* , embedding: NULL }} as r, 
       b {{ .* , embedding: NULL }} as b,
       r.episode_number as episode,
       r.context as description,
       type(r) as relationship_type
ORDER BY r.season, r.episode
""",

    "episode_specific": """
MATCH (a)-[r {{episode_number: '{episode_number}'}}]-(b)
RETURN a {{ .* , embedding: NULL }} as a, 
       r {{ .* , embedding: NULL }} as r, 
       b {{ .* , embedding: NULL }} as b,
       r.context as description,
       type(r) as relationship_type
ORDER BY a.name
""",

    "bidirectional_relationship": """
MATCH path = (a {{name: '{entity_a}'}})-[r]-(b {{name: '{entity_b}'}})
WHERE r.episode_number IS NOT NULL
WITH a, r, b, 
     CASE 
       WHEN startNode(r) = a THEN 'outgoing'
       ELSE 'incoming'
     END as direction
RETURN a {{ .* , embedding: NULL }} as a, 
       r {{ .* , embedding: NULL }} as r, 
       b {{ .* , embedding: NULL }} as b,
       r.episode_number as episode,
       r.context as description,
       type(r) as relationship_type,
       direction
ORDER BY r.season, r.episode
"""
}

# ============================================================
# 예시 쿼리 (Few-shot Learning)
# ============================================================
# ============================================================================
# 2. 쿼리 예제 (Query Examples)
# ============================================================================

EXAMPLES_SINGLE_CHARACTER = [
    {
        "category": "단일 캐릭터 활약",
        "user_input": "카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?",
        "cypher": """
MATCH (n:인간 {name: '카마도 탄지로'})-[r]-(m)
WHERE r.episode_number IS NOT NULL AND r.season = 1
RETURN n { .* , embedding: NULL } as n, 
       r { .* , embedding: NULL } as r, 
       m { .* , embedding: NULL } as m, 
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS rel_type
ORDER BY r.episode
        """.strip()
    },
    {
        "category": "역할 분석",
        "user_input": "토미오카 기유는 어떤 역할을 했는지 에피소드별로 알려줘",
        "cypher": """
MATCH (n:인간 {name: '토미오카 기유'})-[r]-(m)
WHERE r.episode_number IS NOT NULL
RETURN n { .* , embedding: NULL } as n, 
       r { .* , embedding: NULL } as r, 
       m { .* , embedding: NULL } as m,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS rel_type
ORDER BY r.season, r.episode
        """.strip()
    }
]

EXAMPLES_RELATIONSHIPS = [
    {
        "category": "두 캐릭터 관계",
        "user_input": "카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어?",
        "cypher": """
MATCH (a:인간 {name: '카마도 탄지로'})-[r]-(b:인간 {name: '카마도 네즈코'})
WHERE r.episode_number IS NOT NULL
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS rel_type
ORDER BY r.season, r.episode
        """.strip()
    },
    {
        "category": "동료 관계",
        "user_input": "아가츠마 젠이츠와 하시비라 이노스케의 관계는?",
        "cypher": """
MATCH (a:인간 {name: '아가츠마 젠이츠'})-[r]-(b:인간 {name: '하시비라 이노스케'})
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       type(r) AS rel_type
ORDER BY r.season, r.episode
        """.strip()
    }
]

EXAMPLES_BATTLES = [
    {
        "category": "특정 전투",
        "user_input": "카마도 탄지로가 루이와 싸운 에피소드는?",
        "cypher": """
MATCH (a:인간 {name: '카마도 탄지로'})-[r:FIGHTS|BATTLES]-(b:도깨비 {name: '루이'})
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       r.outcome AS result
ORDER BY r.season, r.episode
        """.strip()
    },
    {
        "category": "모든 전투",
        "user_input": "카마도 탄지로의 모든 전투 기록을 보여줘",
        "cypher": """
MATCH (a:인간 {name: '카마도 탄지로'})-[r:FIGHTS|BATTLES|DEFEATS]-(b:도깨비)
RETURN a { .* , embedding: NULL } as a, 
       r { .* , embedding: NULL } as r, 
       b { .* , embedding: NULL } as b,
       r.episode_number AS episode, 
       r.context AS description, 
       r.outcome AS result
ORDER BY r.season, r.episode
        """.strip()
    }
]

# 모든 예제 통합
ALL_EXAMPLES = EXAMPLES_SINGLE_CHARACTER + EXAMPLES_RELATIONSHIPS + EXAMPLES_BATTLES

# 프롬프트용 예제 문자열 생성
def format_examples_for_prompt(examples_list):
    """예제를 프롬프트 형식으로 변환"""
    formatted = []
    for ex in examples_list:
        formatted.append(
            f"# {ex['category']}\n"
            f"USER INPUT: '{ex['user_input']}'\n"
            f"QUERY:\n{ex['cypher']}\n"
        )
    return "\n".join(formatted)

FORMATTED_EXAMPLES = format_examples_for_prompt(ALL_EXAMPLES)

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
7. **임베딩 제외**: RETURN 시 { .* , embedding: NULL } 구문 사용하여 임베딩 속성 제외

## 자주 쓰는 패턴:
- 단일 캐릭터: MATCH (n:인간 {name: '이름'})-[r]-(m)
- 두 캐릭터: MATCH (a {name: '이름1'})-[r]-(b {name: '이름2'})
- 전투만: -[r:FIGHTS|BATTLES|DEFEATS]-
- 시즌 필터: WHERE r.season = 1
"""


# ============================================================
# 개선된 프롬프트 템플릿
# ============================================================
# ============================================================================
# 4. 최종 프롬프트 템플릿 (Final Prompt Template)
# ============================================================================

SYSTEM_PROMPT = f"""당신은 Neo4j Cypher 쿼리 전문가입니다.
사용자의 한국어 질문을 분석하여 정확한 Cypher 쿼리를 생성하세요.

{{FULL_SCHEMA}}

{{CYPHER_RULES}}
"""

MAIN_PROMPT_TEMPLATE = """
## 예제 (Examples):
{examples}

## 사용자 질문:
{query_text}

## 지침:
1. 위 스키마와 예제를 참고하여 Cypher 쿼리만 생성하세요
2. 설명이나 주석 없이 실행 가능한 쿼리만 출력하세요
3. 속성은 단일 중괄호 {{}} 사용하세요
4. 캐릭터 이름은 정확히 매칭하세요
5. **반환 값에서 반드시 embedding 속성을 제외하세요**: `node {{ .* , embedding: NULL }}`

OUTPUT (Cypher 쿼리만):
"""

# ============================================================================
# 5. 사용 예제 (Usage Example)
# ============================================================================

def create_text2cypher_prompt(user_query: str, include_all_examples: bool = True):
    """
    Text2Cypher 프롬프트 생성
    
    Args:
        user_query: 사용자 질문
        include_all_examples: 모든 예제 포함 여부
    
    Returns:
        완성된 프롬프트 문자열
    """
    examples = FORMATTED_EXAMPLES if include_all_examples else format_examples_for_prompt(EXAMPLES_SINGLE_CHARACTER[:1])
    
    # SYSTEM_PROMPT의 변수들은 이미 채워져 있어야 함
    return SYSTEM_PROMPT.format(FULL_SCHEMA=FULL_SCHEMA, CYPHER_RULES=CYPHER_RULES) + "\n" + MAIN_PROMPT_TEMPLATE.format(
        examples=examples,
        query_text=user_query
    )

# ============================================================
# 질문 유형 분류기
# ============================================================
def classify_query_type(question: str) -> Tuple[QueryType, Dict[str, str]]:
    """
    질문 유형을 분류하고 엔티티를 추출

    Returns:
        (QueryType, entities_dict)
    """
    # 이름 패턴 매칭
    name_pattern = r'(카마도 탄지로|카마도 네즈코|토미오카 기유|우로코다키 사콘지|사비토|마코모|아가츠마 젠이츠|하시비라 이노스케|츠유리 카나오|렌고쿠 쿄쥬로|우부야시키 카가야|코쵸우 시노부|시나즈가와 사네미|키부츠지 무잔|스사마루|야하바|쿄우가이|루이|엔무)'

    names_found = re.findall(name_pattern, question)

    # 에피소드 패턴 매칭
    episode_pattern = r'(S\d+E\d+|시즌\s*\d+\s*에피소드\s*\d+|제\s*\d+화)'
    episode_match = re.search(episode_pattern, question)

    entities = {}

    # 특정 에피소드 질문
    if episode_match:
        entities['episode'] = episode_match.group(0)
        return QueryType.EPISODE_SPECIFIC, entities

    # 두 인물 간 관계 질문
    if len(names_found) >= 2:
        entities['entity_a'] = names_found[0]
        entities['entity_b'] = names_found[1]
        # "사이에", "간", "관계" 등의 키워드 확인
        if any(keyword in question for keyword in ['사이에', '간', '관계', '와']):
            return QueryType.RELATIONSHIP, entities

    # 단일 인물 질문
    if len(names_found) >= 1:
        entities['entity_name'] = names_found[0]
        return QueryType.SINGLE_ENTITY, entities

    # 일반 질문
    return QueryType.GENERAL, entities

# ============================================================
# Cypher 쿼리 검증기
# ============================================================
class CypherValidator:
    """Cypher 쿼리 유효성 검사"""

    @staticmethod
    def validate_syntax(query: str) -> Tuple[bool, Optional[str]]:
        """기본 구문 검증"""
        query = query.strip()

        # MATCH가 있어야 함
        if not re.search(r'MATCH', query, re.IGNORECASE):
            return False, "MATCH 절이 없습니다"

        # RETURN이 있어야 함
        if not re.search(r'RETURN', query, re.IGNORECASE):
            return False, "RETURN 절이 없습니다"

        # 괄호 균형 확인
        if query.count('(') != query.count(')'):
            return False, "괄호가 균형을 이루지 않습니다"

        # 중괄호 균형 확인
        if query.count('{') != query.count('}'):
            return False, "중괄호가 균형을 이루지 않습니다"

        return True, None

    @staticmethod
    def validate_schema(query: str) -> Tuple[bool, Optional[str]]:
        """스키마 정합성 검증"""
        # 유효한 라벨 확인
        labels = re.findall(r':(\w+)', query)
        valid_labels = {'인간', '도깨비'}
        for label in labels:
            if label not in valid_labels:
                return False, f"잘못된 라벨: {label}"

        return True, None

    @staticmethod
    def test_query(driver, query: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """실제 쿼리 실행 테스트"""
        try:
            with driver.session() as session:
                result = session.run(query)
                records = list(result)
                return True, None, records
        except Exception as e:
            return False, str(e), None

# ============================================================
# 개선된 리트리버 래퍼
# ============================================================
class ImprovedText2CypherRetriever:
    """다단계 검증 및 재시도를 지원하는 리트리버"""

    def __init__(self, driver, llm, examples, schema, max_retries=3):
        self.driver = driver
        self.llm = llm
        self.examples = examples
        self.schema = schema
        self.max_retries = max_retries
        self.validator = CypherValidator()

    def _get_cypher_template(self, query_type: QueryType, entities: Dict[str, str]) -> str:
        """질문 유형에 맞는 Cypher 템플릿 반환"""
        if query_type == QueryType.RELATIONSHIP and 'entity_a' in entities and 'entity_b' in entities:
            return QUERY_TEMPLATES["relationship_between_two"].format(**entities)
        elif query_type == QueryType.SINGLE_ENTITY and 'entity_name' in entities:
            return QUERY_TEMPLATES["single_entity_timeline"].format(**entities)
        elif query_type == QueryType.EPISODE_SPECIFIC and 'episode' in entities:
            return QUERY_TEMPLATES["episode_specific"].format(**entities)
        else:
            # 일반 질문은 LLM에게 맡김
            return None

    def search(self, query_text: str) -> Dict[str, Any]:
        """
        검색 실행 (다단계 검증 및 재시도)

        Returns:
            {
                'success': bool,
                'cypher': str,
                'items': list,
                'error': str (optional)
            }
        """
        # 질문 유형 분류
        query_type, entities = classify_query_type(query_text)
        print(f"\n🔍 질문 유형: {query_type.value}")
        print(f"📝 추출된 엔티티: {entities}")

        # 템플릿 기반 쿼리 시도
        template_query = self._get_cypher_template(query_type, entities)

        if template_query:
            # 템플릿이 있으면 직접 실행
            print(f"\n📋 템플릿 기반 Cypher:\n{template_query}")
            
            try:
                test_valid, test_error, records = self.validator.test_query(
                    self.driver, template_query
                )
                
                if test_valid and records and len(records) > 0:
                    print(f"  ✅ 템플릿 쿼리 성공: {len(records)}개 레코드 발견")
                    
                    # RetrieverResult 형식으로 변환
                    items = []
                    for record in records:
                        # 레코드를 문자열로 변환
                        content = str(dict(record))
                        items.append(RetrieverResultItem(content=content))
                    
                    return {
                        'success': True,
                        'cypher': template_query,
                        'items': items,
                        'metadata': {'cypher': template_query}
                    }
                else:
                    print(f"  ⚠️  템플릿 쿼리 결과 없음, LLM 생성으로 전환")
            except Exception as e:
                print(f"  ⚠️  템플릿 실행 실패: {e}, LLM 생성으로 전환")

        # LLM 기반 쿼리 생성 시도
        feedback = ""

        for attempt in range(self.max_retries):
            print(f"\n🔄 시도 {attempt + 1}/{self.max_retries}")

            try:
                # 프롬프트 빌드 (새로운 함수 사용)
                prompt = create_text2cypher_prompt(query_text + (f"\n\n[피드백]: {feedback}" if feedback else ""), include_all_examples=True)

                # LLM 호출하여 Cypher 생성
                start_time = time.time()
                response = self.llm.invoke(prompt)
                elapsed_time = time.time() - start_time
                print(f"  ⏱️  Cypher generation completed in {elapsed_time:.2f}s")
                cypher_query = response.content.strip()
                
                # 코드 블록 제거 및 설명문 제거
                cypher_query = re.sub(r'```cypher\n?', '', cypher_query)
                cypher_query = re.sub(r'```\n?', '', cypher_query)
                
                # 설명문 제거 (한국어 설명이 포함된 경우)
                # "... Cypher 쿼리입니다:" 다음의 MATCH 문만 추출
                match_pattern = re.search(r'(MATCH\s+.*)', cypher_query, re.DOTALL | re.IGNORECASE)
                if match_pattern:
                    cypher_query = match_pattern.group(1)
                
                cypher_query = cypher_query.strip()

                print(f"📋 생성된 Cypher:\n{cypher_query}")

                # 1. 구문 검증
                syntax_valid, syntax_error = self.validator.validate_syntax(cypher_query)
                if not syntax_valid:
                    feedback = f"구문 오류: {syntax_error}"
                    print(f"  ❌ {feedback}")
                    continue

                # 2. 스키마 검증
                schema_valid, schema_error = self.validator.validate_schema(cypher_query)
                if not schema_valid:
                    feedback = f"스키마 오류: {schema_error}"
                    print(f"  ❌ {feedback}")
                    continue

                # 3. 실행 테스트
                test_valid, test_error, records = self.validator.test_query(
                    self.driver, cypher_query
                )
                if not test_valid:
                    feedback = f"실행 오류: {test_error}"
                    print(f"  ❌ {feedback}")
                    continue

                # 4. 결과 확인
                if not records or len(records) == 0:
                    feedback = "결과 없음. 관계 방향 또는 이름 매칭을 재검토하세요. 무방향 패턴 (a)-[r]-(b) 사용을 권장합니다."
                    print(f"  ⚠️  {feedback}")
                    # 결과가 없어도 성공으로 처리 (데이터가 실제로 없을 수 있음)
                    if attempt == self.max_retries - 1:
                        return {
                            'success': True,
                            'cypher': cypher_query,
                            'items': [],
                            'metadata': {'cypher': cypher_query}
                        }
                    continue

                # 성공
                print(f"  ✅ 검증 통과: {len(records)}개 레코드 발견")
                
                # RetrieverResult 형식으로 변환
                items = []
                for record in records:
                    content = str(dict(record))
                    items.append(RetrieverResultItem(content=content))
                
                return {
                    'success': True,
                    'cypher': cypher_query,
                    'items': items,
                    'metadata': {'cypher': cypher_query}
                }

            except Exception as e:
                feedback = f"예외 발생: {str(e)}"
                print(f"  ❌ {feedback}")
                if attempt == self.max_retries - 1:
                    return {
                        'success': False,
                        'cypher': cypher_query if 'cypher_query' in locals() else None,
                        'items': [],
                        'error': feedback
                    }
                continue

        # 모든 시도 실패
        return {
            'success': False,
            'cypher': None,
            'items': [],
            'error': f"최대 재시도 횟수 초과. 마지막 오류: {feedback}"
        }

# ============================================================
# 리트리버 초기화
# ============================================================
retriever = ImprovedText2CypherRetriever(
    driver=driver,
    llm=llm,
    examples=ALL_EXAMPLES,  # 리스트 전달 (ImprovedText2CypherRetriever 내부 수정 필요)
    schema=FULL_SCHEMA,
    max_retries=4
)
# ============================================================
# 하이브리드 리트리버 (Vector Search + Cypher Traversal)
# ============================================================
class HybridRetriever:
    """벡터 검색과 Cypher 그래프 순회를 결합한 하이브리드 검색"""
    
    def __init__(
        self, 
        driver, 
        llm, 
        embedder, 
        cypher_retriever: ImprovedText2CypherRetriever,
        top_k: int = 5,
        expansion_depth: int = 2
    ):
        self.driver = driver
        self.llm = llm
        self.embedder = embedder
        self.cypher_retriever = cypher_retriever
        self.top_k = top_k
        self.expansion_depth = expansion_depth
    
    def vector_search(self, query_text: str) -> List[Dict[str, Any]]:
        """벡터 인덱스를 사용한 의미적 유사도 검색 (레이블별 인덱스 쿼리)"""
        try:
            # 1. 질문 임베딩
            print(f"\n🔍 Vector search for: '{query_text}'")
            start_time = time.time()
            query_embedding = self.embedder.embed_query(query_text)
            elapsed = time.time() - start_time
            print(f"  ⏱️  Query embedding generated in {elapsed:.2f}s")
            
            # 2. 각 레이블별 벡터 인덱스 쿼리 (2_ingest_data_v2.py에서 생성한 인덱스 구조와 일치)
            entity_labels = ["인간", "도깨비"]
            all_seed_nodes = []
            
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                for label in entity_labels:
                    index_name = f"entity_embeddings_{label}"
                    try:
                        result = session.run(
                            f"""
                            CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding)
                            YIELD node, score
                            RETURN node.id as id, node.name as name, labels(node) as labels, score
                            ORDER BY score DESC
                            """,
                            top_k=self.top_k,
                            embedding=query_embedding
                        )
                        
                        for record in result:
                            all_seed_nodes.append({
                                'id': record['id'],
                                'name': record['name'],
                                'labels': record['labels'],
                                'score': record['score']
                            })
                    except Exception as e:
                        print(f"  ⚠️  Failed to query index '{index_name}': {e}")
                
                # 3. 스코어 기준으로 정렬하고 top_k만 선택
                all_seed_nodes.sort(key=lambda x: x['score'], reverse=True)
                seed_nodes = all_seed_nodes[:self.top_k]
                
                print(f"  ✅ Found {len(seed_nodes)} seed nodes via vector search (from {len(all_seed_nodes)} total)")
                for node in seed_nodes:
                    print(f"    - {node['name']} (score: {node['score']:.4f})")
                
                return seed_nodes
                
        except Exception as e:
            print(f"  ⚠️  Vector search failed: {e}")
            return []
    
    def expand_from_seeds(self, seed_node_ids: List[str]) -> List[Dict[str, Any]]:
        """시드 노드로부터 그래프 확장"""
        if not seed_node_ids:
            return []
        
        try:
            print(f"\n🌐 Expanding graph from {len(seed_node_ids)} seed nodes...")
            
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                # 시드 노드로부터 1-2 홉 이웃 탐색
                # Use f-string for depth since parameters can't be used in path patterns
                result = session.run(
                    f"""
                    MATCH (seed)
                    WHERE seed.id IN $seed_ids
                    MATCH path = (seed)-[r*1..{self.expansion_depth}]-(neighbor)
                    WHERE r[0].episode_number IS NOT NULL
                    WITH seed, neighbor, relationships(path) as rels, length(path) as dist
                    UNWIND rels as rel
                    RETURN DISTINCT 
                        seed.name as seed_name,
                        neighbor.name as neighbor_name,
                        type(rel) as rel_type,
                        rel.episode_number as episode,
                        rel.context as context,
                        dist as distance
                    ORDER BY dist, episode
                    LIMIT 50
                    """,
                    seed_ids=seed_node_ids
                )
                
                expanded_results = []
                for record in result:
                    expanded_results.append(dict(record))
                
                print(f"  ✅ Expanded to {len(expanded_results)} relationships")
                return expanded_results
                
        except Exception as e:
            print(f"  ⚠️  Graph expansion failed: {e}")
            return []
    
    def search(self, query_text: str) -> Dict[str, Any]:
        """하이브리드 검색 실행"""
        print("\n" + "="*80)
        print("🔀 HYBRID RETRIEVAL: Vector Search + Cypher Traversal")
        print("="*80)
        
        # 1. 벡터 검색으로 시드 노드 찾기
        seed_nodes = self.vector_search(query_text)
        
        hybrid_context = []
        
        if seed_nodes:
            # 2. 그래프 확장
            seed_ids = [node['id'] for node in seed_nodes]
            expanded_results = self.expand_from_seeds(seed_ids)
            
            if expanded_results:
                # 벡터 검색 결과를 컨텍스트로 변환
                for item in expanded_results:
                    context_str = (
                        f"[{item.get('episode', 'N/A')}] "
                        f"{item.get('seed_name', '')} "
                        f"--[{item.get('rel_type', '')}]--> "
                        f"{item.get('neighbor_name', '')}: "
                        f"{item.get('context', '')}"
                    )
                    hybrid_context.append(RetrieverResultItem(content=context_str))
                
                print(f"\n✅ Hybrid search found {len(hybrid_context)} context items from vector expansion")
        
        # 3. Cypher 쿼리 실행 (폴백 또는 보완)
        print("\n" + "-"*80)
        print("🔍 Running Cypher query for additional context...")
        print("-"*80)
        
        cypher_result = self.cypher_retriever.search(query_text)
        
        # 4. 결과 병합
        if cypher_result.get('success'):
            cypher_items = cypher_result.get('items', [])
            
            # 하이브리드 컨텍스트와 Cypher 결과 결합
            all_items = hybrid_context + cypher_items
            
            # 중복 제거 (간단한 문자열 비교)
            seen = set()
            unique_items = []
            for item in all_items:
                content_str = str(item.content)
                if content_str not in seen:
                    seen.add(content_str)
                    unique_items.append(item)
            
            print(f"\n📊 Total unique items: {len(unique_items)} (vector: {len(hybrid_context)}, cypher: {len(cypher_items)})")
            
            return {
                'success': True,
                'cypher': cypher_result.get('cypher'),
                'items': unique_items,
                'metadata': {
                    'cypher': cypher_result.get('cypher'),
                    'vector_seed_count': len(seed_nodes),
                    'vector_context_count': len(hybrid_context),
                    'cypher_context_count': len(cypher_items),
                    'total_unique_count': len(unique_items)
                }
            }
        else:
            # Cypher 실패 시 벡터 결과만 사용
            if hybrid_context:
                print("\n⚠️  Cypher query failed, using only vector search results")
                return {
                    'success': True,
                    'cypher': None,
                    'items': hybrid_context,
                    'metadata': {
                        'vector_only': True,
                        'vector_context_count': len(hybrid_context)
                    }
                }
            else:
                return cypher_result  # 둘 다 실패


# 컨텍스트 정제 및 확장


# Initialize hybrid retriever if enabled
if config.USE_HYBRID_RETRIEVAL:
    hybrid_retriever = HybridRetriever(
        driver=driver,
        llm=llm,
        embedder=embedder,
        cypher_retriever=retriever,
        top_k=config.VECTOR_TOP_K,
        expansion_depth=config.GRAPH_EXPANSION_DEPTH
    )
    print("✅ Hybrid retrieval enabled (vector search + Cypher traversal)")
else:
    hybrid_retriever = None
    print("ℹ️  Using pure Cypher retrieval (hybrid disabled)")

# ============================================================
# ============================================================
def clean_context(raw_content: str) -> str:
    """컨텍스트 정제 (element_id 등 제거)"""
    # element_id 제거
    cleaned = re.sub(r"element_id='[^']*'\s*", "", raw_content)
    # labels=frozenset 제거
    cleaned = re.sub(r"labels=frozenset\([^)]*\)\s*", "", cleaned)
    # nodes=\(<Node...>, <Node...>\) 단순화
    cleaned = re.sub(r"nodes=\([^)]*\)", "", cleaned)
    return cleaned.strip()

def has_json_artifacts(text: str) -> bool:
    """JSON 파싱 오류 흔적 감지"""
    artifacts = [
        r'}}+,',
        r'제외하세요',
        r'포함하세요',
        r'JSON 포맷',
        r'gave you',
    ]
    for pattern in artifacts:
        if re.search(pattern, text):
            return True
    return False

def filter_and_clean_results(result_items) -> List[str]:
    """결과 필터링 및 정제"""
    cleaned_contexts = []

    for item in result_items:
        raw = str(item.content)

        # JSON 아티팩트가 있으면 제외
        if has_json_artifacts(raw):
            print(f"  ⚠️  JSON 아티팩트 감지로 제외: {raw[:100]}...")
            continue

        # 정제
        cleaned = clean_context(raw)
        if cleaned:
            cleaned_contexts.append(cleaned)

    return cleaned_contexts

# ============================================================
# 개선된 답변 생성 프롬프트
# ============================================================
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

# ============================================================
# LLM 호출 함수
# ============================================================
def llm_call(prompt: str) -> str:
    """LLM을 호출하여 답변 생성"""
    start_time = time.time()
    response = llm.invoke(prompt)
    elapsed_time = time.time() - start_time
    print(f"⏱️  Answer generation completed in {elapsed_time:.2f}s")
    return response.content

# ============================================================
# 메인 GraphRAG 파이프라인
# ============================================================
def graphrag_pipeline(user_question: str) -> str:
    """
    개선된 GraphRAG 파이프라인

    Args:
        user_question: 사용자 질문

    Returns:
        최종 답변 문자열
    """
    print("\n" + "="*100)
    print(f"❓ 사용자 질문: {user_question}")
    print("="*100)

    # 1. 검색 실행 (하이브리드 또는 순수 Cypher)
    try:
        if config.USE_HYBRID_RETRIEVAL and hybrid_retriever:
            search_result = hybrid_retriever.search(query_text=user_question)
        else:
            search_result = retriever.search(query_text=user_question)
    except Exception as e:
        traceback.print_exc()
        return f"❌ 검색 중 오류가 발생했습니다: {e}"

    # 2. 검색 실패 처리
    if not search_result.get('success', False):
        error_msg = search_result.get('error', '알 수 없는 오류')
        return f"❌ 검색에 실패했습니다: {error_msg}"

    # 3. Cypher 쿼리 확인
    cypher_query = search_result.get('cypher')
    print(f"\n✅ 최종 Cypher 쿼리:\n{cypher_query}")

    # 4. 결과 확인
    result_items = search_result.get('items', [])
    print(f"\n📊 검색된 레코드 수: {len(result_items)}")

    if not result_items:
        return "❌ 데이터베이스에서 관련 정보를 찾을 수 없습니다. 캐릭터 이름이나 질문을 다시 확인해주세요."

    # 5. 컨텍스트 정제 및 필터링
    cleaned_contexts = filter_and_clean_results(result_items)

    if not cleaned_contexts:
        return "⚠️  검색 결과가 있으나 유효한 정보를 추출할 수 없습니다."

    print(f"\n✅ 정제된 컨텍스트 수: {len(cleaned_contexts)}")

    # 6. 컨텍스트 조합
    full_context = "\n\n".join(cleaned_contexts)

    # 7. 답변 생성 프롬프트 구성
    final_prompt = ANSWER_GENERATION_PROMPT.format(
        question=user_question,
        context=full_context
    )

    # 8. 최종 답변 생성
    print(f"\n🤖 답변 생성 중... (Prompt length: {len(final_prompt)})")
    final_answer = llm_call(final_prompt)

    return final_answer

# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    # 테스트 질문들
    queries = [
        "카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어? 에피소드별로 정리해줘.",
        # "토미오카 기유는 시즌 1에서 어떤 역할을 했는지 에피소드별로 알려줘.",
        # "카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?",
        # "아가츠마 젠이츠와 하시비라 이노스케는 언제 처음 만났어?",
    ]

    for i, query in enumerate(tqdm(queries, desc="Processing queries", unit="query"), 1):
        print(f"\n\n{'#'*100}")
        print(f"# 테스트 {i}/{len(queries)}")
        print(f"{'#'*100}")

        try:
            answer = graphrag_pipeline(query)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            print(f"\n❌ [Error] LLM 연결 오류 발생: {e}")
            print("💡 팁: Docker 컨테이너(Ollama)가 실행 중인지, 또는 모델이 로딩 중인지 확인하세요.")
            print("   (Dockers 폴더에서 'docker-compose up -d' 실행 필요)")
            answer = "오류 발생: AI 모델에 연결할 수 없습니다."
        except Exception as e:
            print(f"\n❌ [Error] 예상치 못한 오류 발생: {e}")
            traceback.print_exc()
            answer = "오류 발생: 시스템 내부 오류입니다."

        print(f"\n{'='*100}")
        print("📝 최종 답변:")
        print("="*100)
        print(answer)
        print(f"\n{'='*100}\n")

        # 다음 질문 전 구분선
        if i < len(queries):
            print("\n" + "⏸"*50 + "\n")
