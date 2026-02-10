import json
import re
import os
from typing import List, Dict, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

import config

# OpenAI 클라이언트 초기화
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

# 타입 정의
PropertyValue = Union[str, int, float, bool, None]

# ============================================================
# 유효한 노드 정의 (마스터 데이터)
# ============================================================
VALID_NODES = {
    "N0":  {"label": "인간", "name": "카마도 탄지로"},
    "N1":  {"label": "인간", "name": "카마도 네즈코"},
    "N2":  {"label": "인간", "name": "토미오카 기유"},
    "N3":  {"label": "인간", "name": "우로코다키 사콘지"},
    "N4":  {"label": "인간", "name": "사비토"},
    "N5":  {"label": "인간", "name": "마코모"},
    "N6":  {"label": "인간", "name": "아가츠마 젠이츠"},
    "N7":  {"label": "인간", "name": "하시비라 이노스케"},
    "N8":  {"label": "인간", "name": "츠유리 카나오"},
    "N9":  {"label": "인간", "name": "렌고쿠 쿄쥬로"},
    "N10": {"label": "인간", "name": "우부야시키 카가야"},
    "N11": {"label": "인간", "name": "코쵸우 시노부"},
    "N12": {"label": "인간", "name": "시나즈가와 사네미"},
    "N13": {"label": "도깨비", "name": "키부츠지 무잔"},
    "N14": {"label": "도깨비", "name": "스사마루"},
    "N15": {"label": "도깨비", "name": "야하바"},
    "N16": {"label": "도깨비", "name": "쿄우가이"},
    "N17": {"label": "도깨비", "name": "루이"},
    "N18": {"label": "도깨비", "name": "엔무"},
}

# 유효한 관계 타입 정의 (개선: 더 구체적인 관계 추가)
VALID_RELATIONSHIP_TYPES = [
    "FIGHTS",           # 싸움
    "PROTECTS",         # 보호
    "TRAINS",           # 훈련
    "TRAINS_WITH",      # 함께 훈련
    "KNOWS",            # 알고 있음
    "FAMILY_OF",        # 가족
    "SIBLING_OF",       # 형제/자매
    "ALLY_OF",          # 동맹
    "ENEMY_OF",         # 적
    "DEFEATS",          # 물리침
    "SAVES",            # 구함
    "RESCUES",          # 구출
    "MEETS",            # 만남
    "ENCOUNTERS",       # 조우
    "GUIDES",           # 안내
    "ATTACKS",          # 공격
    "DEFENDS",          # 방어
    "TRANSFORMS",       # 변신
    "JOINS",            # 합류
    "SUPPORTS",         # 지원
    "REUNITES_WITH",    # 재회
    "BATTLES",          # 전투
    "HEALS",            # 치료
    "TEACHES",          # 가르침
]

# 영어 → 한국어 이름 매핑
ENGLISH_TO_KOREAN_NAME = {
    "Tanjiro Kamado": "카마도 탄지로",
    "Tanjiro": "카마도 탄지로",
    "Nezuko Kamado": "카마도 네즈코",
    "Nezuko": "카마도 네즈코",
    "Giyu Tomioka": "토미오카 기유",
    "Giyu": "토미오카 기유",
    "Sakonji Urokodaki": "우로코다키 사콘지",
    "Sakonji": "우로코다키 사콘지",
    "Sabito": "사비토",
    "Makomo": "마코모",
    "Zenitsu Agatsuma": "아가츠마 젠이츠",
    "Zenitsu": "아가츠마 젠이츠",
    "Inosuke Hashibira": "하시비라 이노스케",
    "Inosuke": "하시비라 이노스케",
    "Kanao Tsuyuri": "츠유리 카나오",
    "Kanao": "츠유리 카나오",
    "Kyojuro Rengoku": "렌고쿠 쿄쥬로",
    "Rengoku": "렌고쿠 쿄쥬로",
    "Kagaya Ubuyashiki": "우부야시키 카가야",
    "Kagaya": "우부야시키 카가야",
    "Shinobu Kocho": "코쵸우 시노부",
    "Shinobu": "코쵸우 시노부",
    "Sanemi Shinazugawa": "시나즈가와 사네미",
    "Sanemi": "시나즈가와 사네미",
    "Muzan Kibutsuji": "키부츠지 무잔",
    "Muzan": "키부츠지 무잔",
    "Susamaru": "스사마루",
    "Yahaba": "야하바",
    "Kyogai": "쿄우가이",
    "Rui": "루이",
    "Enmu": "엔무",
}

# 이름 → ID 역매핑
NAME_TO_ID = {v["name"]: k for k, v in VALID_NODES.items()}
ENGLISH_NAME_TO_ID = {
    eng_name: NAME_TO_ID.get(ENGLISH_TO_KOREAN_NAME[eng_name])
    for eng_name in ENGLISH_TO_KOREAN_NAME
    if ENGLISH_TO_KOREAN_NAME[eng_name] in NAME_TO_ID
}

class Node(BaseModel):
    id: str
    label: str
    properties: Optional[Dict[str, PropertyValue]] = None

class Relationship(BaseModel):
    type: str
    start_node_id: str
    end_node_id: str
    properties: Optional[Dict[str, PropertyValue]] = None

class GraphResponse(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

# ============================================================
# 개선된 프롬프트 템플릿 (Phase 1 개선)
# ============================================================
EXTRACTION_PROMPT_TEMPLATE = """당신은 애니메이션 "귀멸의 칼날"의 지식 그래프를 구축하는 전문가입니다.
주어진 에피소드 시놉시스에서 등장인물 간의 관계를 정확히 추출하세요.

## 🎯 핵심 규칙:
1. **엔티티 추출**: 아래 VALID_NODES에 정의된 캐릭터만 사용
2. **정확한 이름 매칭**: 영어 이름은 ENGLISH_TO_KOREAN 매핑 참조
3. **노드 ID 형식**: 반드시 "N0" ~ "N18" 중 하나만 사용
4. **관계 타입**: VALID_RELATIONSHIP_TYPES 중에서만 선택
5. **에피소드 정보**: 각 관계의 맥락을 간결하게 설명 (50자 이내)
6. **출력 형식**: 유효한 JSON만 반환 (설명문, 주석, 추가 텍스트 금지)

## 📋 VALID_NODES (사용 가능한 캐릭터):
{valid_nodes_json}

## 🔗 VALID_RELATIONSHIP_TYPES:
{valid_relationship_types}

## 🌐 영어-한국어 이름 매핑:
{english_to_korean_mapping}

## 📝 출력 JSON 형식 (반드시 준수):
{{
  "nodes": [
    {{
      "id": "N0",
      "label": "인간",
      "properties": {{"name": "카마도 탄지로"}}
    }}
  ],
  "relationships": [
    {{
      "type": "PROTECTS",
      "start_node_id": "N0",
      "end_node_id": "N1",
      "properties": {{
        "description": "관계의 상세 맥락 설명 (선택 사항, 최대 100자)",
        "outcome": "전투/상호작용 결과 (예: victory, defeat, 승리, 패배)",
        "context": "관계가 형성된 배경 (예: 도깨비로 변한 네즈코를 보호하기로 결심)",
        "action": "수행된 구체적 행동 (예: kill Father, reaffirm faith)",
        "role": "관계에서의 역할 (예: defender of spirits, 주요 가족 구성원)",
        "technique": "사용된 기술/능력 (예: Blood Demon Art, Hinokami Kagura)",
        "method": "사용된 방법/수단 (예: water breathing antidote, night-long ritualistic dance)",
        "effectiveness": "효과 정도 (예: high, medium, low)",
        "duration": "지속 기간 (예: short-term, long-term)",
        "event": "관련 이벤트 (예: train departure, boarding, afterfall)",
        "assistant": "보조자 이름 (해당 시)",
        "protectees": "보호 대상 (해당 시)",
        "commendation": "칭찬/인정 내용 (해당 시)",
        "subject": "훈련/교육 대상 (해당 시)",
        "enemy": "적대 대상 이름 (해당 시)",
        "from": "출발/시작 인물",
        "to": "도착/목표 인물",
        
        "NOTE": "위 필드는 가이드입니다. 관계 타입에 따라 적절한 필드만 선택하여 사용하세요. 반드시 모든 필드를 포함할 필요는 없습니다."
      }}
    }}
  ]
}}

## 📋 Properties 사용 가이드 (관계 타입별):

### FIGHTS / BATTLES / DEFEATS:
- **필수**: outcome (승리/패배 결과)
- **선택**: technique (사용 기술), enemy (적 이름), description (전투 상세)

### PROTECTS / SAVES / RESCUES:
- **필수**: description 또는 action (보호/구출 행위)
- **선택**: protectees (보호 대상), effectiveness (효과), duration (지속기간), method (방법)

### TRAINS / TEACHES:
- **필수**: subject (훈련/교육 대상)
- **선택**: assistant (보조자), description (훈련 내용), method (훈련 방식)

### MEETS / ENCOUNTERS:
- **필수**: context 또는 event (만남의 배경/이벤트)
- **선택**: outcome (만남의 결과), commendation (평가/반응)

### KNOWS / ALLY_OF / ENEMY_OF:
- **필수**: description 또는 context (관계의 맥락)
- **선택**: from/to (관계의 방향성), role (역할)

### 기타 관계 (JOINS, SUPPORTS, HEALS 등):
- **필수**: context 또는 description (관계의 기본 맥락)
- **선택**: 상황에 맞는 추가 필드

**💡 중요**: 위 가이드는 권장사항입니다. 시놉시스에 명시된 정보만 사용하고, 정보가 없는 필드는 포함하지 마세요.

## 입력 시놉시스:
에피소드: S{season}E{episode:02d}
{synopsis}

## ⚠️ 주의사항:
- 추측하지 말고 시놉시스에 명시된 내용만 추출
- 동일 관계 중복 추출 금지
- JSON 외 다른 텍스트 포함 금지

## JSON 응답:"""

def build_extraction_prompt(synopsis: str, season: int, episode: int) -> str:
    """개선된 추출 프롬프트 생성"""
    valid_nodes_json = json.dumps(
        [{"id": k, "label": v["label"], "name": v["name"]} for k, v in VALID_NODES.items()],
        ensure_ascii=False, indent=2
    )
    valid_relationship_types = ", ".join(VALID_RELATIONSHIP_TYPES)
    english_to_korean_mapping = json.dumps(ENGLISH_TO_KOREAN_NAME, ensure_ascii=False, indent=2)

    return EXTRACTION_PROMPT_TEMPLATE.format(
        valid_nodes_json=valid_nodes_json,
        valid_relationship_types=valid_relationship_types,
        english_to_korean_mapping=english_to_korean_mapping,
        season=season,
        episode=episode,
        synopsis=synopsis
    )

def llm_call_structured(prompt: str, model: str = config.LLM_MODEL, max_retries: int = 3) -> Optional[GraphResponse]:
    """구조화된 출력으로 OpenAI API 호출 (재시도 로직 추가)"""
    for attempt in range(max_retries):
        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=GraphResponse,
                temperature=0.1,  # 일관성을 위해 낮은 temperature
            )
            return resp.choices[0].message.parsed
        except Exception as e:
            print(f"  ⚠️ API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            continue
    return None

def normalize_node_id(node_id: str) -> Optional[str]:
    """노드 ID 정규화 - 유효하지 않으면 None 반환"""
    if not node_id:
        return None

    # 불필요한 문자 제거
    cleaned = re.sub(r'[^N0-9]', '', str(node_id).upper())

    # N + 숫자 형식 추출
    match = re.match(r'(N\d+)', cleaned)
    if match:
        normalized = match.group(1)
        if normalized in VALID_NODES:
            return normalized
    return None

def normalize_relationship_type(rel_type: str) -> Optional[str]:
    """관계 타입 정규화 (개선: 더 정교한 매핑)"""
    if not rel_type:
        return "KNOWS"  # 기본값

    # 대문자로 변환하고 불필요한 문자 제거
    cleaned = re.sub(r'[^A-Z_]', '', str(rel_type).upper())

    # 유효한 관계 타입이면 반환
    if cleaned in VALID_RELATIONSHIP_TYPES:
        return cleaned

    # 개선: 더 정교한 유사 타입 매핑
    type_mapping = {
        "FIGHT": "FIGHTS",
        "BATTLE": "BATTLES",
        "PROTECT": "PROTECTS",
        "TRAIN": "TRAINS",
        "KNOW": "KNOWS",
        "FAMILY": "FAMILY_OF",
        "SIBLING": "SIBLING_OF",
        "ALLY": "ALLY_OF",
        "ENEMY": "ENEMY_OF",
        "DEFEAT": "DEFEATS",
        "SAVE": "SAVES",
        "RESCUE": "RESCUES",
        "MEET": "MEETS",
        "ENCOUNTER": "ENCOUNTERS",
        "GUIDE": "GUIDES",
        "ATTACK": "ATTACKS",
        "DEFEND": "DEFENDS",
        "TRANSFORM": "TRANSFORMS",
        "JOIN": "JOINS",
        "SUPPORT": "SUPPORTS",
        "REUNITE": "REUNITES_WITH",
        "HEAL": "HEALS",
        "TEACH": "TEACHES",
        "DEFENDS_FROM": "DEFENDS",
        "SAVES_FROM": "SAVES",
        "PROTECTED_BY": "PROTECTS",
        "TRAINED_BY": "TRAINS",
    }

    return type_mapping.get(cleaned, "KNOWS")

def has_json_artifacts(text: str) -> bool:
    """JSON 파싱 오류 흔적 감지"""
    if not text:
        return False

    artifacts = [
        r'}}+,',  # }},
        r'\{[^}]+gave you',  # "gave you" 같은 프롬프트 누출
        r'제외하세요',
        r'포함하세요',
        r'JSON 포맷',
        r'다른 텍스트를 포함하지',
    ]

    for pattern in artifacts:
        if re.search(pattern, text):
            return True
    return False

def clean_property_value(value: PropertyValue) -> PropertyValue:
    """속성 값 정제 (JSON 아티팩트 제거)"""
    if not isinstance(value, str):
        return value

    # JSON 아티팩트가 있으면 None 반환
    if has_json_artifacts(value):
        return None

    # 200자 초과 시 잘라내기
    if len(value) > 200:
        return value[:197] + "..."

    return value.strip()

def validate_and_normalize_node(node: Node) -> Optional[Node]:
    """노드 유효성 검사 및 정규화"""
    # ID 정규화
    normalized_id = normalize_node_id(node.id)
    if not normalized_id:
        return None

    # 마스터 데이터에서 올바른 값 가져오기
    master_node = VALID_NODES[normalized_id]

    return Node(
        id=normalized_id,
        label=master_node["label"],
        properties={"name": master_node["name"]}
    )

def validate_and_normalize_relationship(rel: Relationship) -> Optional[Relationship]:
    """관계 유효성 검사 및 정규화 (개선: 속성 정제 추가)"""
    # 노드 ID 정규화
    start_id = normalize_node_id(rel.start_node_id)
    end_id = normalize_node_id(rel.end_node_id)

    if not start_id or not end_id:
        return None

    # 자기 자신과의 관계 제거
    if start_id == end_id:
        return None

    # 관계 타입 정규화
    rel_type = normalize_relationship_type(rel.type)

    # 속성 정제
    cleaned_properties = {}
    if rel.properties:
        for key, value in rel.properties.items():
            cleaned_value = clean_property_value(value)
            if cleaned_value is not None:
                cleaned_properties[key] = cleaned_value

    return Relationship(
        type=rel_type,
        start_node_id=start_id,
        end_node_id=end_id,
        properties=cleaned_properties if cleaned_properties else None
    )

def validate_and_clean_graph(graph: GraphResponse) -> GraphResponse:
    """그래프 데이터 유효성 검사 및 정제"""
    valid_nodes = []
    valid_relationships = []
    seen_node_ids = set()
    seen_relationships = set()

    # 노드 정제
    for node in graph.nodes:
        normalized = validate_and_normalize_node(node)
        if normalized and normalized.id not in seen_node_ids:
            valid_nodes.append(normalized)
            seen_node_ids.add(normalized.id)

    # 관계 정제
    for rel in graph.relationships:
        normalized = validate_and_normalize_relationship(rel)
        if normalized:
            # 중복 관계 제거 (같은 타입, 같은 노드 쌍)
            rel_key = (normalized.type, normalized.start_node_id, normalized.end_node_id)
            if rel_key not in seen_relationships:
                valid_relationships.append(normalized)
                seen_relationships.add(rel_key)

    return GraphResponse(nodes=valid_nodes, relationships=valid_relationships)

def combine_chunk_graphs(chunk_graphs: List[GraphResponse]) -> GraphResponse:
    """여러 GraphResponse를 하나로 합칩니다."""
    all_nodes = []
    all_relationships = []
    seen_nodes = set()
    seen_relationships = set()

    for chunk_graph in chunk_graphs:
        # 노드 수집 (중복 제거)
        for node in chunk_graph.nodes:
            node_key = node.id
            if node_key not in seen_nodes:
                all_nodes.append(node)
                seen_nodes.add(node_key)

        # 관계 수집 (개선: 에피소드별 동일 관계 허용)
        for rel in chunk_graph.relationships:
            rel_key = (
                rel.type,
                rel.start_node_id,
                rel.end_node_id,
                rel.properties.get("episode_number") if rel.properties else None
            )
            if rel_key not in seen_relationships:
                all_relationships.append(rel)
                seen_relationships.add(rel_key)

    return GraphResponse(nodes=all_nodes, relationships=all_relationships)

def fetch_episode(link: str) -> List[dict]:
    """위키피디아에서 에피소드 데이터를 가져옵니다"""
    season = int(re.search(r"season_(\d+)", link).group(1))
    print(f"📥 Season {season} 데이터 가져오는 중: {link}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(link, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")

    episodes = []
    rows = table.select("tr.vevent.module-episode-list-row")

    for i, row in enumerate(rows, start=1):
        synopsis = None
        synopsis_row = row.find_next_sibling("tr", class_="expand-child")
        if synopsis_row:
            synopsis_cell = synopsis_row.select_one("td.description div.shortSummaryText")
            synopsis = synopsis_cell.get_text(strip=True) if synopsis_cell else None

        episodes.append({
            "season": season,
            "episode_in_season": i,
            "synopsis": synopsis,
        })

    return episodes

def collect_data(use_cache: bool = True) -> List[dict]:
    """여러 시즌에서 에피소드 데이터를 수집합니다"""
    print("\n" + "="*60)
    print("📚 데이터 수집 시작")
    print("="*60)

    cache_file = "output/raw_data_v3.json"

    # 캐시 파일이 있으면 먼저 사용
    if use_cache and os.path.exists(cache_file):
        print(f"💾 캐시 파일 사용: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            episodes = json.load(f)
        print(f"✅ 총 {len(episodes)}개 에피소드 로드 완료 (캐시)")
        return episodes

    episode_links = [
        "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
        # "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_2",  # 귀멸의 칼날 시즌 2
    ]

    all_episodes = []
    for link in episode_links:
        try:
            episodes = fetch_episode(link)
            all_episodes.extend(episodes)
        except Exception as e:
            print(f"❌ 데이터 가져오기 실패 ({link}): {e}")
            continue

    print(f"✅ 총 {len(all_episodes)}개 에피소드 수집 완료")
    return all_episodes

def process_data(episodes: List[dict]) -> GraphResponse:
    """에피소드 데이터를 지식 그래프로 처리합니다 (개선: 에러 핸들링 강화)"""
    print("\n" + "="*60)
    print("🔄 데이터 처리 시작")
    print("="*60)

    chunk_graphs: List[GraphResponse] = []
    failed_episodes = []

    for episode in episodes:
        if not episode.get("synopsis"):
            print(f"⏭️  S{episode['season']}E{episode['episode_in_season']:02d}: 시놉시스 없음 - 건너뜀")
            continue

        episode_code = f"S{episode['season']}E{episode['episode_in_season']:02d}"
        print(f"\n🎬 처리 중: {episode_code}")

        try:
            # (1) 개선된 프롬프트 생성 및 LLM 호출
            prompt = build_extraction_prompt(
                episode['synopsis'],
                episode['season'],
                episode['episode_in_season']
            )
            graph_response = llm_call_structured(prompt)

            if graph_response is None:
                print(f"  ❌ LLM 호출 실패")
                failed_episodes.append(episode_code)
                continue

            # (2) 데이터 유효성 검사 및 정제
            graph_response = validate_and_clean_graph(graph_response)

            # (3) 관계에 구조화된 에피소드 정보 추가
            for relationship in graph_response.relationships:
                if relationship.properties is None:
                    relationship.properties = {}
                relationship.properties["episode_number"] = episode_code
                relationship.properties["season"] = episode['season']
                relationship.properties["episode"] = episode['episode_in_season']

            chunk_graphs.append(graph_response)
            print(f"  ✅ 추출: 노드 {len(graph_response.nodes)}개, 관계 {len(graph_response.relationships)}개")

        except Exception as e:
            print(f"  ❌ 처리 중 오류: {e}")
            failed_episodes.append(episode_code)
            continue

    if not chunk_graphs:
        raise Exception("❌ 그래프를 성공적으로 추출하지 못했습니다.")

    print(f"\n{'='*60}")
    print(f"✅ 처리 완료: {len(chunk_graphs)}개 에피소드 성공")
    if failed_episodes:
        print(f"⚠️  실패한 에피소드: {', '.join(failed_episodes)}")
    print("="*60)

    return combine_chunk_graphs(chunk_graphs)

def save_output(episodes: List[dict], final_graph: GraphResponse):
    """출력을 JSON 파일로 저장합니다"""
    print("\n" + "="*60)
    print("💾 결과 저장")
    print("="*60)

    os.makedirs("output", exist_ok=True)

    # 원본 데이터 저장
    with open("output/raw_data_v3.json", "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print("✅ 원본 데이터: output/raw_data_v3.json")

    # 최종 지식 그래프 저장
    with open("output/knowledge_graph_v3.json", "w", encoding="utf-8") as f:
        json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)
    print("✅ 최종 지식그래프: output/knowledge_graph_v3.json")

    # 통계 정보 저장
    stats = generate_statistics(final_graph)
    with open("output/statistics_v3.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("✅ 통계 정보: output/statistics_v3.json")

def generate_statistics(graph: GraphResponse) -> Dict[str, Any]:
    """지식 그래프 통계 생성"""
    # 노드별 관계 수 계산
    node_degree = {}
    for rel in graph.relationships:
        node_degree[rel.start_node_id] = node_degree.get(rel.start_node_id, 0) + 1
        node_degree[rel.end_node_id] = node_degree.get(rel.end_node_id, 0) + 1

    # 관계 타입별 빈도
    rel_type_count = {}
    for rel in graph.relationships:
        rel_type_count[rel.type] = rel_type_count.get(rel.type, 0) + 1

    # 에피소드별 관계 수
    episode_count = {}
    for rel in graph.relationships:
        if rel.properties and "episode_number" in rel.properties:
            ep = rel.properties["episode_number"]
            episode_count[ep] = episode_count.get(ep, 0) + 1

    return {
        "total_nodes": len(graph.nodes),
        "total_relationships": len(graph.relationships),
        "node_degree": dict(sorted(node_degree.items(), key=lambda x: x[1], reverse=True)),
        "relationship_type_count": dict(sorted(rel_type_count.items(), key=lambda x: x[1], reverse=True)),
        "episode_relationship_count": dict(sorted(episode_count.items())),
    }

def validate_final_output() -> bool:
    """최종 출력 파일 검증 (개선: 더 상세한 검증)"""
    print("\n" + "="*60)
    print("🔍 최종 출력 검증")
    print("="*60)

    try:
        with open("output/knowledge_graph_v3.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return False

    errors = []
    warnings = []

    # 노드 검증
    for node in data.get("nodes", []):
        if node["id"] not in VALID_NODES:
            errors.append(f"잘못된 노드 ID: {node['id']}")
        if node["label"] not in ["인간", "도깨비"]:
            errors.append(f"잘못된 라벨: {node['label']} (노드: {node['id']})")
        if not node.get("properties", {}).get("name"):
            warnings.append(f"노드 {node['id']}: 이름 속성 없음")

    # 관계 검증
    for i, rel in enumerate(data.get("relationships", [])):
        if rel["start_node_id"] not in VALID_NODES:
            errors.append(f"관계 {i}: 잘못된 시작 노드 {rel['start_node_id']}")
        if rel["end_node_id"] not in VALID_NODES:
            errors.append(f"관계 {i}: 잘못된 종료 노드 {rel['end_node_id']}")
        if rel["type"] not in VALID_RELATIONSHIP_TYPES:
            errors.append(f"관계 {i}: 잘못된 관계 타입 {rel['type']}")

        # 에피소드 번호 검증
        if rel.get("properties"):
            ep_num = rel["properties"].get("episode_number")
            if ep_num and not re.match(r'S\d+E\d+', ep_num):
                warnings.append(f"관계 {i}: 잘못된 에피소드 형식 {ep_num}")

            # JSON 아티팩트 검증
            for key, value in rel["properties"].items():
                if isinstance(value, str) and has_json_artifacts(value):
                    warnings.append(f"관계 {i}: JSON 아티팩트 감지 in '{key}'")

    # 결과 출력
    if errors:
        print(f"\n❌ 검증 실패: {len(errors)}개 오류 발견")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... 외 {len(errors) - 20}개 오류")
        return False

    if warnings:
        print(f"\n⚠️  경고: {len(warnings)}개")
        for warning in warnings[:10]:
            print(f"  - {warning}")

    print(f"\n✅ 검증 통과!")
    print(f"  📊 유효한 노드: {len(data.get('nodes', []))}개")
    print(f"  🔗 유효한 관계: {len(data.get('relationships', []))}개")

    # 통계 출력
    try:
        with open("output/statistics_v3.json", "r", encoding="utf-8") as f:
            stats = json.load(f)

        print(f"\n📈 상위 연결 노드:")
        for node_id, count in list(stats["node_degree"].items())[:5]:
            node_name = VALID_NODES[node_id]["name"]
            print(f"  - {node_name} ({node_id}): {count}개 관계")

        print(f"\n🔗 상위 관계 타입:")
        for rel_type, count in list(stats["relationship_type_count"].items())[:5]:
            print(f"  - {rel_type}: {count}개")
    except:
        pass

    return True

def main():
    """전체 프로세스를 조율하는 메인 함수"""
    try:
        print("\n" + "="*60)
        print("🚀 개선된 지식그래프 생성기 v3.0")
        print("="*60)

        # 단계 1: 데이터 수집
        episodes = collect_data()

        if not episodes:
            raise Exception("❌ 수집된 에피소드 데이터가 없습니다.")

        # 단계 2: 데이터 처리
        final_graph = process_data(episodes)

        # 단계 3: 출력 저장
        save_output(episodes, final_graph)

        # 단계 4: 검증
        is_valid = validate_final_output()

        print("\n" + "="*60)
        if is_valid:
            print("✅ 지식그래프 생성 완료!")
        else:
            print("⚠️  지식그래프 생성 완료 (일부 검증 실패)")
        print(f"📊 총 노드 수: {len(final_graph.nodes)}")
        print(f"🔗 총 관계 수: {len(final_graph.relationships)}")
        print("\n📁 생성된 파일:")
        print("  - output/raw_data_v3.json")
        print("  - output/knowledge_graph_v3.json")
        print("  - output/statistics_v3.json")
        print("="*60)

        return 0 if is_valid else 1

    except Exception as e:
        print(f"\n❌ 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
