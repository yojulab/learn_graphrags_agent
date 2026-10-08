
import json
import re
import os
from typing import List, Dict, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

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

# LLM 처리용 템플릿
UPDATED_TEMPLATE = """
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Extract the entities (nodes) and specify their type from the following text, but **you MUST select nodes ONLY from the following predefined set** (see the provided NODES list below). Do not create any new nodes or use names that do not exactly match one in the NODES list.

Also extract the relationships between these nodes. Return the result as JSON using the following format:

{
  "nodes": [
    {"id": "N0", "label": "인간", "properties": {"name": "Tanjiro Kamado"}}
  ],
  "relationships": [
    {"type": "FIGHTS", "start_node_id": "N0", "end_node_id": "N13", "properties": {"outcome": "victory"}}
  ]
}

Additional rules:
- Use only nodes from the NODES list. Do not invent or substitute nodes.
- Skip any relationship if one of its entities is not in NODES.
- Only output valid relationships where both endpoints exist in NODES and the direction matches their types.

NODES =
[
  {"id":"N0",  "label":"인간", "properties":{"name":"Tanjiro Kamado"}},
  {"id":"N1",  "label":"인간", "properties":{"name":"Nezuko Kamado"}},
  {"id":"N2",  "label":"인간", "properties":{"name":"Giyu Tomioka"}},
  {"id":"N3",  "label":"인간", "properties":{"name":"Sakonji Urokodaki"}},
  {"id":"N4",  "label":"인간", "properties":{"name":"Sabito"}},
  {"id":"N5",  "label":"인간", "properties":{"name":"Makomo"}},
  {"id":"N6",  "label":"인간", "properties":{"name":"Zenitsu Agatsuma"}},
  {"id":"N7",  "label":"인간", "properties":{"name":"Inosuke Hashibira"}},
  {"id":"N8",  "label":"인간", "properties":{"name":"Kanao Tsuyuri"}},
  {"id":"N9",  "label":"인간", "properties":{"name":"Kyojuro Rengoku"}},
  {"id":"N10", "label":"인간", "properties":{"name":"Kagaya Ubuyashiki"}},
  {"id":"N11", "label":"인간", "properties":{"name":"Shinobu Kocho"}},
  {"id":"N12", "label":"인간", "properties":{"name":"Sanemi Shinazugawa"}},
  {"id":"N13", "label":"도깨비", "properties":{"name":"Muzan Kibutsuji"}},
  {"id":"N14", "label":"도깨비", "properties":{"name":"Susamaru"}},
  {"id":"N15", "label":"도깨비", "properties":{"name":"Yahaba"}},
  {"id":"N16", "label":"도깨비", "properties":{"name":"Kyogai"}},
  {"id":"N17", "label":"도깨비", "properties":{"name":"Rui"}},
  {"id":"N18", "label":"도깨비", "properties":{"name":"Enmu"}}
]
"""

# 한국어 노드 이름 매핑
# 노드 이름 한글 매핑 (귀살대 · 도깨비)
KOREAN_NODE_MAP = {
    # 귀살대 (귀살대)
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

    # 도깨비 (도깨비)
    "Muzan Kibutsuji": "키부츠지 무잔",
    "Susamaru": "스사마루",
    "Yahaba": "야하바",
    "Kyogai": "쿄우가이",
    "Rui": "루이",
    "Enmu": "엔무",
}


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

def combine_chunk_graphs(chunk_graphs: list) -> 'GraphResponse':
    """
    여러 개의 GraphResponse 객체를 하나로 합칩니다.
    - 모든 노드와 관계(relationship)를 모읍니다.
    - 중복된 노드는 제거하고, 처음 등장한 노드만 남깁니다.
    """
    # 1. 모든 chunk_graph에서 노드를 수집합니다
    all_nodes = []
    for chunk_graph in chunk_graphs:
        for node in chunk_graph.nodes:
            all_nodes.append(node)
    
    # 2. 모든 chunk_graph에서 관계(relationship)를 수집합니다
    all_relationships = []
    for chunk_graph in chunk_graphs:
        for relationship in chunk_graph.relationships:
            all_relationships.append(relationship)
    
    # 3. 중복된 노드를 제거합니다
    unique_nodes = []
    seen = set()  # 이미 추가된 노드를 기억해둘 집합

    for node in all_nodes:
        # 노드의 id, label, properties를 묶어서 하나의 키로 만듭니다
        node_key = (node.id, node.label, str(node.properties))
        # 이미 추가된 노드가 아니라면 unique_nodes에 추가합니다
        if node_key not in seen:
            unique_nodes.append(node)
            seen.add(node_key)

    # 4. 중복이 제거된 노드들과 모든 관계를 합쳐 새로운 GraphResponse를 만듭니다
    return GraphResponse(nodes=unique_nodes, relationships=all_relationships)

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

def collect_data() -> List[dict]:
    """여러 시즌에서 에피소드 데이터를 수집합니다"""
    print("=== 데이터 수집 시작 ===")
    
    # Check for existing raw data
    raw_data_path = "output/raw_data.json"
    if os.path.exists(raw_data_path):
        print(f"기존 데이터 발견: {raw_data_path}")
        with open(raw_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    episode_links = [
        "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",  # 귀멸의 칼날 시즌 1
        # 필요에 따라 더 많은 시즌 추가:
        # "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_2",  # 귀멸의 칼날 시즌 2
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
            # (1) 노드 표준화를 위한 업데이트된 템플릿으로 프롬프트 생성
            prompt = UPDATED_TEMPLATE + f"\n 입력값\n {episode['synopsis']}"
            graph_response = llm_call_structured(prompt)

            # (2) 관계에 에피소드 번호 추가 (예: S1E01)
            episode_number = f"S{episode['season']}E{episode['episode_in_season']:02d}"

            for relationship in graph_response.relationships:
                if relationship.properties is None:
                    relationship.properties = {}
                relationship.properties["episode_number"] = episode_number
                
            # (3) 노드 이름을 한국어로 변환
            for node in graph_response.nodes:
                english_name = node.properties.get("name", "")
                if english_name in KOREAN_NODE_MAP:
                    node.properties["name"] = KOREAN_NODE_MAP[english_name]
            
            chunk_graphs.append(graph_response)
            
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
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs("output", exist_ok=True)
    
    # 원본 데이터 저장
    with open("output/raw_data_v1.json", "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print("원본 데이터 저장: output/raw_data_v1.json")
    
    # 최종 지식 그래프 저장
    with open("output/knowledge_graph_v1.json", "w", encoding="utf-8") as f:
        json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)
    print("최종 지식그래프 저장: output/knowledge_graph_v1.json")

def main():
    """전체 프로세스를 조율하는 메인 함수"""
    try:
        # config.py에서 로드된 값을 확인 (선택 사항)
        # if not config.OPENAI_API_KEY: ...
        
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
        
        print("=" * 50)
        print("✅ 지식그래프 생성 완료!")
        print(f"📊 총 노드 수: {len(final_graph.nodes)}")
        print(f"🔗 총 관계 수: {len(final_graph.relationships)}")
        print("생성된 파일:")
        print("- output/raw_data_v1.json")
        print("- output/knowledge_graph_v1.json")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
