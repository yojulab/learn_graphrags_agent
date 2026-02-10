"""
개선된 GraphRAG 에이전트 v2.0
- 다단계 쿼리 검증 및 재시도
- 질문 유형 분류 및 템플릿 기반 쿼리 생성
- 컨텍스트 확장 전략
- 개선된 프롬프트 템플릿
"""

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import RetrieverResultItem
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
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
        # <think>...</think> 블록 제거
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # 잔여 태그 제거
        content = re.sub(r'.*?</think>', '', content, flags=re.DOTALL)
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
    base_url=config.MODEL_API_URL
)

# ============================================================
# 스키마 정의
# ============================================================
SCHEMA = """
## 노드 라벨:
- 인간: 인간 캐릭터 (예: 카마도 탄지로, 카마도 네즈코)
- 도깨비: 도깨비 캐릭터 (예: 키부츠지 무잔, 루이)

## 관계 타입:
- FIGHTS: 싸움 (예: 탄지로가 무잔과 싸움)
- PROTECTS: 보호 (예: 탄지로가 네즈코를 보호)
- TRAINS: 훈련 (예: 사콘지가 탄지로를 훈련)
- TRAINS_WITH: 함께 훈련
- SIBLING_OF: 형제/자매 관계
- FAMILY_OF: 가족 관계
- ALLY_OF: 동맹 관계
- ENEMY_OF: 적 관계
- DEFEATS: 물리침
- SAVES: 구함
- RESCUES: 구출
- MEETS: 만남
- ENCOUNTERS: 조우
- GUIDES: 안내
- ATTACKS: 공격
- DEFENDS: 방어
- SUPPORTS: 지원
- REUNITES_WITH: 재회
- HEALS: 치료
- TEACHES: 가르침
- BATTLES: 전투
- JOINS: 합류
- TRANSFORMS: 변신

## 노드 속성:
- id: 노드 고유 ID (N0, N1, ...)
- name: 캐릭터 이름 (예: '카마도 탄지로')

## 관계 속성:
- episode_number: 에피소드 번호 (형식: S1E01, S1E02, ...)
- season: 시즌 번호 (정수)
- episode: 에피소드 번호 (정수)
- context: 사건 설명 (문자열)
- outcome: 결과 (예: '승리', '패배', '도주')

## 주요 캐릭터 목록:
인간: 카마도 탄지로, 카마도 네즈코, 토미오카 기유, 우로코다키 사콘지, 사비토, 마코모, 
      아가츠마 젠이츠, 하시비라 이노스케, 츠유리 카나오, 렌고쿠 쿄쥬로, 
      우부야시키 카가야, 코쵸우 시노부, 시나즈가와 사네미
도깨비: 키부츠지 무잔, 스사마루, 야하바, 쿄우가이, 루이, 엔무
"""

# ============================================================
# 쿼리 템플릿 정의 (개선)
# ============================================================
QUERY_TEMPLATES = {
    "single_entity_timeline": """
MATCH (entity {{name: '{entity_name}'}})-[r]-(other)
WHERE r.episode_number IS NOT NULL
RETURN entity, r, other, 
       r.episode_number as episode, 
       r.context as description,
       type(r) as relationship_type,
       labels(other) as other_labels
ORDER BY r.season, r.episode
""",

    "relationship_between_two": """
MATCH (a {{name: '{entity_a}'}})-[r]-(b {{name: '{entity_b}'}})
WHERE r.episode_number IS NOT NULL
RETURN a, r, b,
       r.episode_number as episode,
       r.context as description,
       type(r) as relationship_type
ORDER BY r.season, r.episode
""",

    "episode_specific": """
MATCH (a)-[r {{episode_number: '{episode_number}'}}]-(b)
RETURN a, r, b,
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
RETURN a, r, b,
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
EXAMPLES = [
    # 단일 인물 질문
    "USER INPUT: '카마도 탄지로는 시즌 1에서 에피소드별로 어떤 활약을 했어?' "
    "QUERY: MATCH (n {name: '카마도 탄지로'})-[r]-(m) "
    "WHERE r.episode_number IS NOT NULL "
    "RETURN n, r, m, r.episode_number as episode, r.context as description, type(r) as rel_type "
    "ORDER BY r.season, r.episode",

    # 두 인물 간 관계
    "USER INPUT: '카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어?' "
    "QUERY: MATCH (a {name: '카마도 탄지로'})-[r]-(b {name: '카마도 네즈코'}) "
    "WHERE r.episode_number IS NOT NULL "
    "RETURN a, r, b, r.episode_number as episode, r.context as description, type(r) as rel_type "
    "ORDER BY r.season, r.episode",

    # 역할 질문
    "USER INPUT: '토미오카 기유는 시즌 1에서 어떤 역할을 했는지 에피소드별로 알려줘.' "
    "QUERY: MATCH (n {name: '토미오카 기유'})-[r]-(m) "
    "WHERE r.episode_number IS NOT NULL "
    "RETURN n, r, m, r.episode_number as episode, r.context as description, type(r) as rel_type "
    "ORDER BY r.season, r.episode",

    # 특정 적과의 전투
    "USER INPUT: '카마도 탄지로가 루이와 싸운 에피소드는?' "
    "QUERY: MATCH (a {name: '카마도 탄지로'})-[r:FIGHTS|BATTLES]-(b {name: '루이'}) "
    "WHERE r.episode_number IS NOT NULL "
    "RETURN a, r, b, r.episode_number as episode, r.context as description "
    "ORDER BY r.season, r.episode",
]

# ============================================================
# 개선된 프롬프트 템플릿
# ============================================================
CYPHER_GENERATION_PROMPT = """당신은 Neo4j Cypher 쿼리 전문가입니다.
사용자의 한국어 질문을 분석하여 정확한 Cypher 쿼리를 생성하세요.

## 데이터베이스 스키마:
{schema}

## 중요 규칙:
1. **정확한 이름 매칭**: 
   - 'name' 속성으로 노드를 찾을 때 정확한 전체 이름 사용 (예: '카마도 탄지로', '카마도 네즈코')
   - 성과 이름을 모두 포함해야 함

2. **에피소드 정렬**: 
   - 시간 순서가 중요한 질문에는 반드시 ORDER BY r.season, r.episode 추가
   - WHERE r.episode_number IS NOT NULL로 에피소드 정보가 있는 관계만 필터링

3. **관계 방향**:
   - 방향이 중요하지 않으면 무방향 패턴 사용: (a)-[r]-(b)
   - 특정 관계 타입이 필요하면: (a)-[r:FIGHTS|PROTECTS]-(b)

4. **필수 반환 값**:
   - 항상 노드(a, b)와 관계(r) 반환
   - 에피소드 정보: r.episode_number as episode
   - 설명: r.context as description
   - 관계 타입: type(r) as relationship_type

5. **출력 형식**:
   - Cypher 쿼리만 출력 (설명 금지)
   - 한 줄로 작성하거나 가독성 있게 여러 줄로 작성

## 예시:
{examples}

## 사용자 질문:
{query_text}

## Cypher 쿼리:"""

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
                # 프롬프트 빌드 (schema와 examples 포함)
                examples_text = "\n".join(self.examples)
                prompt = CYPHER_GENERATION_PROMPT.format(
                    schema=self.schema,
                    examples=examples_text,
                    query_text=query_text + (f"\n\n[피드백]: {feedback}" if feedback else "")
                )

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
    examples=EXAMPLES,
    schema=SCHEMA,
    max_retries=4  # 최대 4회 시도
)

# ============================================================
# 컨텍스트 정제 및 확장
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
아래 데이터베이스 검색 결과를 바탕으로 사용자의 질문에 정확하고 자연스럽게 답변하세요.

## 사용자 질문:
{question}

## 데이터베이스 검색 결과:
{context}

## 답변 작성 규칙:

### 1. 에피소드별 정리 (시간순 질문인 경우)
- **S1E01**: [사건 요약] 형식으로 작성
- 에피소드 번호 순서대로 나열
- 각 에피소드마다 간결하게 1-2문장으로 요약

### 2. 관계 표현 자연화
- DB의 관계명을 그대로 쓰지 말 것:
  ❌ "PROTECTS 관계가 있습니다"
  ✅ "탄지로는 네즈코를 보호했습니다"

- 관계 타입별 자연스러운 표현:
  - PROTECTS → "보호하다", "지키다"
  - FIGHTS/BATTLES → "싸우다", "전투하다"
  - SAVES/RESCUES → "구하다", "구출하다"
  - TRAINS → "훈련시키다", "가르치다"
  - MEETS/ENCOUNTERS → "만나다", "조우하다"
  - DEFEATS → "물리치다", "이기다"
  - REUNITES_WITH → "재회하다"
  - SUPPORTS → "돕다", "지원하다"

### 3. 스토리텔링
- 마치 이야기를 들려주듯 자연스럽게 작성
- 캐릭터의 감정이나 상황을 함께 언급
- 단순 나열보다는 맥락 있는 서술

### 4. 데이터 정확성
- DB 결과에 없는 내용은 추측하지 말 것
- 검색 결과가 부족하면 "제한된 정보로는..." 명시
- 에피소드 번호나 캐릭터 이름은 정확히 표기

### 5. 답변 구조 (권장)
```
[간단한 서론 1문장]

## 에피소드별 주요 사건

- **S1E01**: ...
- **S1E02**: ...
...

[간단한 마무리 1문장 - 선택사항]
```

## ⚠️ 주의사항:
- 기술적 용어(노드, 관계, 프로퍼티 등) 사용 금지
- "데이터베이스에서", "검색 결과에 따르면" 같은 표현 지양
- 자연스러운 한국어로 작성

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

    # 1. 검색 실행 (다단계 검증 포함)
    try:
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
    print(f"\n🤖 답변 생성 중...")
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

        answer = graphrag_pipeline(query)

        print(f"\n{'='*100}")
        print("📝 최종 답변:")
        print("="*100)
        print(answer)
        print(f"\n{'='*100}\n")

        # 다음 질문 전 구분선
        if i < len(queries):
            print("\n" + "⏸"*50 + "\n")
