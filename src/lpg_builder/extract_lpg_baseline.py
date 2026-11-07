"""
[LPG Step 2-A] 기초 프롬프트 기반 LPG(레이블 지정 속성 그래프) 추출 실습
(src/01_lpg_builder/3_extract_lpg_baseline.py)

- 목적: 원시 텍스트(시놉시스)로부터 LLM 구조화 프롬프트를 통해 기초 Node/Edge/Property 추출
- 결과물: output/knowledge_graph_v1.json
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import openai
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config
from src.lpg_builder.lpg_schema import (
    VALID_NODES,
    ENGLISH_TO_KOREAN_NAME,
    LPGNode,
    LPGEdge,
    LPGGraph
)

client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

class RawNode(BaseModel):
    id: str
    label: str
    properties: Optional[Dict[str, Any]] = None

class RawRelationship(BaseModel):
    type: str
    start_node_id: str
    end_node_id: str
    properties: Optional[Dict[str, Any]] = None

class GraphResponse(BaseModel):
    nodes: List[RawNode]
    relationships: List[RawRelationship]

BASELINE_TEMPLATE = """
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
Extract the entities (nodes) and specify their type from the following text, but **you MUST select nodes ONLY from the following predefined set**:

NODES:
""" + json.dumps([{"id": k, "label": v["label"], "name": v["name"]} for k, v in VALID_NODES.items()], ensure_ascii=False, indent=2) + """

Also extract the relationships between these nodes. Return the result as JSON using the following format:
{
  "nodes": [{"id": "N0", "label": "인간", "properties": {"name": "카마도 탄지로"}}],
  "relationships": [{"type": "FIGHTS", "start_node_id": "N0", "end_node_id": "N13", "properties": {"outcome": "victory"}}]
}

Additional rules:
- Use only nodes from the NODES list. Do not invent or substitute nodes.
- Skip any relationship if one of its entities is not in NODES.
"""

def llm_call_structured(prompt: str) -> GraphResponse:
    resp = client.beta.chat.completions.parse(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=GraphResponse,
    )
    return resp.choices[0].message.parsed

def extract_baseline_lpg(raw_data_path: str = "output/raw_data.json") -> LPGGraph:
    file_path = BASE_DIR / raw_data_path
    if not file_path.exists():
        raise FileNotFoundError(f"원시 데이터 파일이 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    all_nodes_dict: Dict[str, LPGNode] = {}
    all_edges: List[LPGEdge] = []

    print(f"🚀 [Baseline LPG] 총 {len(episodes)}개 에피소드 처리 시작...")

    for ep in episodes:
        synopsis = ep.get("synopsis")
        if not synopsis:
            continue
        ep_code = ep.get("episode_code", f"S{ep.get('season')}E{ep.get('episode_in_season'):02d}")
        print(f"  ⚡ {ep_code} 시놉시스 분석 중...")

        prompt = f"{BASELINE_TEMPLATE}\n\n입력 시놉시스:\n{synopsis}"
        try:
            res = llm_call_structured(prompt)
            for node in res.nodes:
                name = (node.properties or {}).get("name", node.id)
                korean_name = ENGLISH_TO_KOREAN_NAME.get(name, name)
                if node.id not in all_nodes_dict:
                    all_nodes_dict[node.id] = LPGNode(
                        id=node.id,
                        label=node.label if node.label in config.NODE_LABELS else "인간",
                        name=korean_name,
                        properties={"name": korean_name}
                    )

            for rel in res.relationships:
                rel_type = rel.type.strip().upper()
                if rel_type not in config.RELATIONSHIP_TYPES:
                    rel_type = "FIGHTS"
                props = rel.properties or {}
                props["episode_number"] = ep_code
                all_edges.append(LPGEdge(
                    start_id=rel.start_node_id,
                    end_id=rel.end_node_id,
                    type=rel_type,
                    properties=props
                ))
        except Exception as e:
            print(f"  ⚠️ {ep_code} 추출 실패: {e}")

    final_graph = LPGGraph(
        version="lpg-v1.0-baseline",
        nodes=list(all_nodes_dict.values()),
        edges=all_edges,
        metadata={"total_episodes": len(episodes)}
    )

    out_file = BASE_DIR / "output/knowledge_graph_v1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"✅ Baseline LPG 추출 완료: {out_file} (노드 {len(final_graph.nodes)}개, 간선 {len(final_graph.edges)}개)")
    return final_graph

if __name__ == "__main__":
    extract_baseline_lpg()
