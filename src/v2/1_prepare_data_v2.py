
import json
import re
import os
from typing import List, Dict, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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

# 유효한 관계 타입 정의
VALID_RELATIONSHIP_TYPES = [
    "FIGHTS",      # 싸움
    "PROTECTS",    # 보호
    "TRAINS",      # 훈련
    "KNOWS",       # 알고 있음
    "FAMILY_OF",   # 가족
    "ALLY_OF",     # 동맹
    "ENEMY_OF",    # 적
    "DEFEATS",     # 물리침
    "SAVES",       # 구함
    "MEETS",       # 만남
]

# 영어 → 한국어 이름 매핑
ENGLISH_TO_KOREAN_NAME = {
    "Tanjiro Kamado": "카마도 탄지로",
    "Nezuko Kamado": "카마도 네즈코",
    "Giyu Tomioka": "토미오카 기유",
    "Sakonji Urokodaki": "우로코다키 사콘지",
    "Sabito": "사비토",
    "Makomo": "마코모",
    "Zenitsu Agatsuma": "아가츠마 젠이츠",
    "Inosuke Hashibira": "하시비라 이노스케",
    "Kanao Tsuyuri": "츠유리 카나오",
    "Kyojuro Rengoku": "렌고쿠 쿄쥬로",
    "Kagaya Ubuyashiki": "우부야시키 카가야",
    "Shinobu Kocho": "코쵸우 시노부",
    "Sanemi Shinazugawa": "시나즈가와 사네미",
    "Muzan Kibutsuji": "키부츠지 무잔",
    "Susamaru": "스사마루",
    "Yahaba": "야하바",
    "Kyogai": "쿄우가이",
    "Rui": "루이",
    "Enmu": "엔무",
}

# 이름 → ID 역매핑
NAME_TO_ID = {v["name"]: k for k, v in VALID_NODES.items()}
ENGLISH_NAME_TO_ID = {
    "Tanjiro Kamado": "N0",
    "Nezuko Kamado": "N1",
    "Giyu Tomioka": "N2",
    "Sakonji Urokodaki": "N3",
    "Sabito": "N4",
    "Makomo": "N5",
    "Zenitsu Agatsuma": "N6",
    "Inosuke Hashibira": "N7",
    "Kanao Tsuyuri": "N8",
    "Kyojuro Rengoku": "N9",
    "Kagaya Ubuyashiki": "N10",
    "Shinobu Kocho": "N11",
    "Sanemi Shinazugawa": "N12",
    "Muzan Kibutsuji": "N13",
    "Susamaru": "N14",
    "Yahaba": "N15",
    "Kyogai": "N16",
    "Rui": "N17",
    "Enmu": "N18",
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
# 프롬프트 템플릿
# ============================================================
EXTRACTION_PROMPT_TEMPLATE = """당신은 지식 그래프를 구축하기 위해 텍스트에서 정보를 추출하는 전문가입니다.
주어진 텍스트에서 등장인물 간의 관계를 추출하세요.

## 중요 규칙:
1. 반드시 아래 VALID_NODES에 정의된 캐릭터만 사용하세요.
2. 노드 ID는 정확히 "N0", "N1", ... "N18" 형식만 사용하세요.
3. 관계 타입은 반드시 VALID_RELATIONSHIP_TYPES 중 하나만 사용하세요.
4. JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

## VALID_NODES:
{valid_nodes_json}

## VALID_RELATIONSHIP_TYPES:
{valid_relationship_types}

## 출력 형식:
{{
  "nodes": [
    {{"id": "N0", "label": "인간", "properties": {{"name": "카마도 탄지로"}}}}
  ],
  "relationships": [
    {{"type": "FIGHTS", "start_node_id": "N0", "end_node_id": "N13", "properties": {{"outcome": "victory"}}}}
  ]
}}

## 입력 텍스트:
{synopsis}

## JSON 응답:"""


def build_extraction_prompt(synopsis: str) -> str:
    """추출 프롬프트 생성"""
    valid_nodes_json = json.dumps(
        [{"id": k, "label": v["label"], "name": v["name"]} for k, v in VALID_NODES.items()],
        ensure_ascii=False, indent=2
    )
    valid_relationship_types = ", ".join(VALID_RELATIONSHIP_TYPES)
    
    return EXTRACTION_PROMPT_TEMPLATE.format(
        valid_nodes_json=valid_nodes_json,
        valid_relationship_types=valid_relationship_types,
        synopsis=synopsis
    )


def llm_call_structured(prompt: str, model: str = config.LLM_MODEL) -> GraphResponse:
    """구조화된 출력으로 OpenAI API 호출"""
    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=GraphResponse,
    )
    return resp.choices[0].message.parsed


def normalize_node_id(node_id: str) -> Optional[str]:
    """노드 ID 정규화 - 유효하지 않으면 None 반환"""
    # 불필요한 문자 제거
    cleaned = re.sub(r'[^N0-9]', '', node_id)
    
    # N + 숫자 형식 추출
    match = re.match(r'(N\d+)', cleaned)
    if match:
        normalized = match.group(1)
        if normalized in VALID_NODES:
            return normalized
    return None


def normalize_relationship_type(rel_type: str) -> Optional[str]:
    """관계 타입 정규화"""
    # 대문자로 변환하고 불필요한 문자 제거
    cleaned = re.sub(r'[^A-Z_]', '', rel_type.upper())
    
    # 유효한 관계 타입이면 반환
    if cleaned in VALID_RELATIONSHIP_TYPES:
        return cleaned
    
    # 유사한 타입 매핑
    type_mapping = {
        "FIGHT": "FIGHTS",
        "BATTLE": "FIGHTS",
        "PROTECT": "PROTECTS",
        "TRAIN": "TRAINS",
        "KNOW": "KNOWS",
        "FAMILY": "FAMILY_OF",
        "ALLY": "ALLY_OF",
        "ENEMY": "ENEMY_OF",
        "DEFEAT": "DEFEATS",
        "SAVE": "SAVES",
        "MEET": "MEETS",
        "ATTACKS": "FIGHTS",
        "ATTACKED": "FIGHTS",
    }
    
    return type_mapping.get(cleaned, "FIGHTS")  # 기본값: FIGHTS


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
    """관계 유효성 검사 및 정규화"""
    # 노드 ID 정규화
    start_id = normalize_node_id(rel.start_node_id)
    end_id = normalize_node_id(rel.end_node_id)
    
    if not start_id or not end_id:
        return None
    
    # 관계 타입 정규화
    rel_type = normalize_relationship_type(rel.type)
    
    return Relationship(
        type=rel_type,
        start_node_id=start_id,
        end_node_id=end_id,
        properties=rel.properties
    )


def validate_and_clean_graph(graph: GraphResponse) -> GraphResponse:
    """그래프 데이터 유효성 검사 및 정제"""
    valid_nodes = []
    valid_relationships = []
    seen_node_ids = set()
    
    # 노드 정제
    for node in graph.nodes:
        normalized = validate_and_normalize_node(node)
        if normalized and normalized.id not in seen_node_ids:
            valid_nodes.append(normalized)
            seen_node_ids.add(normalized.id)
    
    # 관계 정제
    for rel in graph.relationships:
        normalized = validate_and_normalize_relationship(rel)
        if normalized and normalized.start_node_id in VALID_NODES and normalized.end_node_id in VALID_NODES:
            valid_relationships.append(normalized)
    
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
        
        # 관계 수집 (중복 제거)
        for rel in chunk_graph.relationships:
            rel_key = (rel.type, rel.start_node_id, rel.end_node_id, 
                      rel.properties.get("episode_number") if rel.properties else None)
            if rel_key not in seen_relationships:
                all_relationships.append(rel)
                seen_relationships.add(rel_key)
    
    return GraphResponse(nodes=all_nodes, relationships=all_relationships)


def fetch_episode(link: str) -> List[dict]:
    """위키피디아에서 에피소드 데이터를 가져옵니다"""
    season = int(re.search(r"season_(\d+)", link).group(1))
    print(f"Fetching Season {season} from: {link}")
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
    print("=== 데이터 수집 시작 ===")

    # Check for existing raw data (priority)
    raw_data_path = "output/raw_data.json"
    if os.path.exists(raw_data_path):
        print(f"기존 데이터 발견 (공통): {raw_data_path}")
        with open(raw_data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    cache_file = "output/raw_data_v2.json"
    
    # 캐시 파일이 있으면 먼저 사용
    if use_cache and os.path.exists(cache_file):
        print(f"캐시 파일 사용: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            episodes = json.load(f)
        print(f"총 {len(episodes)}개 에피소드 로드 완료 (캐시)")
        return episodes
    
    episode_links = [
        "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
    ]
    
    all_episodes = []
    for link in episode_links:
        try:
            episodes = fetch_episode(link)
            all_episodes.extend(episodes)
        except Exception as e:
            print(f"Error fetching data from {link}: {e}")
            continue
    
    print(f"총 {len(all_episodes)}개 에피소드 수집 완료")
    return all_episodes


def process_data(episodes: List[dict]) -> GraphResponse:
    """에피소드 데이터를 지식 그래프로 처리합니다"""
    print("=== 데이터 처리 시작 ===")
    
    chunk_graphs: List[GraphResponse] = []
    
    for episode in episodes:
        if not episode.get("synopsis"):
            print(f"에피소드 S{episode['season']}E{episode['episode_in_season']:02d}: 시놉시스가 없어 건너뜀")
            continue
            
        print(f"에피소드 처리 중: 시즌 {episode['season']}, 에피소드 {episode['episode_in_season']}")
        
        try:
            # (1) 프롬프트 생성 및 LLM 호출
            prompt = build_extraction_prompt(episode['synopsis'])
            graph_response = llm_call_structured(prompt)
            
            # (2) 데이터 유효성 검사 및 정제
            graph_response = validate_and_clean_graph(graph_response)

            # (3) 관계에 에피소드 번호 추가
            episode_number = f"S{episode['season']}E{episode['episode_in_season']:02d}"
            for relationship in graph_response.relationships:
                if relationship.properties is None:
                    relationship.properties = {}
                relationship.properties["episode_number"] = episode_number
            
            chunk_graphs.append(graph_response)
            print(f"  - 추출된 노드: {len(graph_response.nodes)}, 관계: {len(graph_response.relationships)}")
            
        except Exception as e:
            print(f"  - 에피소드 처리 중 오류 발생: {e}")
            continue
    
    if not chunk_graphs:
        raise Exception("그래프를 성공적으로 추출하지 못했습니다.")
    
    print(f"총 {len(chunk_graphs)}개 에피소드 처리 완료")
    return combine_chunk_graphs(chunk_graphs)


def save_output(episodes: List[dict], final_graph: GraphResponse):
    """출력을 JSON 파일로 저장합니다"""
    print("=== 결과 저장 ===")
    
    os.makedirs("output", exist_ok=True)
    
    # 원본 데이터 저장
    with open("output/raw_data_v2.json", "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print("원본 데이터 저장: output/raw_data_v2.json")
    
    # 최종 지식 그래프 저장
    with open("output/knowledge_graph_v2.json", "w", encoding="utf-8") as f:
        json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)
    print("최종 지식그래프 저장: output/knowledge_graph_v2.json")


def validate_final_output():
    """최종 출력 파일 검증"""
    print("\n=== 최종 출력 검증 ===")
    
    with open("output/knowledge_graph_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    errors = []
    
    # 노드 검증
    for node in data.get("nodes", []):
        if node["id"] not in VALID_NODES:
            errors.append(f"잘못된 노드 ID: {node['id']}")
        if node["label"] not in ["인간", "도깨비"]:
            errors.append(f"잘못된 라벨: {node['label']} (노드: {node['id']})")
    
    # 관계 검증
    for rel in data.get("relationships", []):
        if rel["start_node_id"] not in VALID_NODES:
            errors.append(f"잘못된 시작 노드: {rel['start_node_id']}")
        if rel["end_node_id"] not in VALID_NODES:
            errors.append(f"잘못된 종료 노드: {rel['end_node_id']}")
        if rel["type"] not in VALID_RELATIONSHIP_TYPES:
            errors.append(f"잘못된 관계 타입: {rel['type']}")
    
    if errors:
        print("❌ 검증 실패:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개 오류")
        return False
    
    print("✅ 검증 통과!")
    print(f"  - 유효한 노드: {len(data.get('nodes', []))}개")
    print(f"  - 유효한 관계: {len(data.get('relationships', []))}개")
    return True


def main():
    """전체 프로세스를 조율하는 메인 함수"""
    try:
        print("🚀 지식그래프 생성기 시작")
        print("=" * 50)
        
        # 단계 1: 데이터 수집
        episodes = collect_data()
        
        if not episodes:
            raise Exception("수집된 에피소드 데이터가 없습니다.")
        
        # 단계 2: 데이터 처리
        final_graph = process_data(episodes)
        
        # 단계 3: 출력 저장
        save_output(episodes, final_graph)
        
        # 단계 4: 검증
        validate_final_output()
        
        print("=" * 50)
        print("✅ 지식그래프 생성 완료!")
        print(f"📊 총 노드 수: {len(final_graph.nodes)}")
        print(f"🔗 총 관계 수: {len(final_graph.relationships)}")
        print("생성된 파일:")
        print("- output/raw_data_v2.json")
        print("- output/knowledge_graph_v2.json")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
