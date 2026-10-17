"""
[v4] Apache AGE 호환 지식그래프 데이터 준비 스크립트
(src/v4/1_prepare_data_v4.py)

역할:
1. Pydantic v2 기반 노드(19개, 인간/도깨비) 및 관계(24종 표준 동사) 스키마 검증
2. output/knowledge_graph_v3.json 또는 원시 데이터를 AGE 정규화 형식으로 변환
3. output/knowledge_graph_v4.json 아티팩트 저장 및 무결성 검증
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

# ============================================================
# Pydantic v2 Data Models
# ============================================================

class NodeModel(BaseModel):
    id: str
    label: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        clean = v.strip()
        if clean not in config.NODE_LABELS:
            raise ValueError(f"유효하지 않은 노드 라벨: '{clean}'. 허용 목록: {config.NODE_LABELS}")
        return clean

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("노드 name은 비어 있을 수 없습니다.")
        return v.strip()


class RelationshipModel(BaseModel):
    start_id: str
    end_id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in config.RELATIONSHIP_TYPES:
            raise ValueError(f"유효하지 않은 관계 타입: '{clean}'. 허용 목록: {config.RELATIONSHIP_TYPES}")
        return clean


class KnowledgeGraphV4Model(BaseModel):
    version: str = "v4.0-apache-age"
    nodes: List[NodeModel]
    relationships: List[RelationshipModel]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def prepare_data_v4() -> KnowledgeGraphV4Model:
    """AGE 호환 지식그래프 데이터 정규화 및 검증"""
    print("=" * 80)
    print("🚀 [v4.0] Apache AGE 호환 데이터셋 준비 및 Pydantic v2 정규화 시작")
    print("=" * 80)

    v3_kg_path = BASE_DIR / "output/knowledge_graph_v3.json"
    v4_kg_path = BASE_DIR / "output/knowledge_graph_v4.json"

    if not v3_kg_path.exists():
        raise FileNotFoundError(f"기준 데이터셋이 존재하지 않습니다: {v3_kg_path}")

    with open(v3_kg_path, "r", encoding="utf-8") as f:
        v3_data = json.load(f)

    raw_nodes = v3_data.get("nodes", [])
    raw_rels = v3_data.get("relationships", [])

    print(f"📦 원본 노드: {len(raw_nodes)}개, 원본 관계: {len(raw_rels)}개 로드됨")

    # 1. 노드 정규화 및 모델 바인딩
    validated_nodes: List[NodeModel] = []
    for item in raw_nodes:
        # v3 데이터 호환 변환
        node_id = item.get("id")
        label = item.get("label", "인간")
        props = item.get("properties", {})
        name = props.get("name") or item.get("name") or node_id
        
        # properties에 필수 식별자 병합
        clean_props = {**props, "name": name, "id": node_id}
        node_obj = NodeModel(
            id=node_id,
            label=label,
            name=name,
            properties=clean_props
        )
        validated_nodes.append(node_obj)

    # 2. 관계 정규화 및 모델 바인딩
    validated_rels: List[RelationshipModel] = []
    valid_node_ids = {n.id for n in validated_nodes}

    for item in raw_rels:
        start_id = item.get("start_node_id") or item.get("start_id")
        end_id = item.get("end_node_id") or item.get("end_id")
        rel_type = item.get("type", "").strip().upper()
        props = item.get("properties", {})

        if start_id not in valid_node_ids or end_id not in valid_node_ids:
            print(f"⚠️  알 수 없는 노드 참조 건너뜀: {start_id} -> {end_id}")
            continue

        if rel_type not in config.RELATIONSHIP_TYPES:
            # 24개 표준 관계 중 매핑 불가 시 경고 및 건너뜀
            print(f"⚠️  표준 스키마 외 관계 타입 건너뜀: {rel_type}")
            continue

        raw_ep = props.get("episode_number") or props.get("episode", 1)
        if isinstance(raw_ep, int):
            ep_int = raw_ep
        else:
            ep_match = re.search(r'\d+', str(raw_ep))
            ep_int = int(ep_match.group()) if ep_match else 1

        clean_props = {
            "episode_number": ep_int,
            "season": props.get("season", 1),
            "context": props.get("context") or props.get("description", "")
        }

        rel_obj = RelationshipModel(
            start_id=start_id,
            end_id=end_id,
            type=rel_type,
            properties=clean_props
        )
        validated_rels.append(rel_obj)

    kg_v4 = KnowledgeGraphV4Model(
        version="v4.0-apache-age",
        nodes=validated_nodes,
        relationships=validated_rels,
        metadata={
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "PostgreSQL 16+ & Apache AGE 1.5+",
            "graph_name": config.AGE_GRAPH_NAME,
            "node_count": len(validated_nodes),
            "relationship_count": len(validated_rels),
            "schema_labels": config.NODE_LABELS,
            "relationship_types_count": len(config.RELATIONSHIP_TYPES)
        }
    )

    # 아티팩트 디스크 저장
    v4_kg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v4_kg_path, "w", encoding="utf-8") as f:
        json.dump(kg_v4.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"✅ [v4 데이터 준비 완료] 노드 {len(validated_nodes)}개, 관계 {len(validated_rels)}개")
    print(f"📁 저장 완료: {v4_kg_path}")
    print("=" * 80 + "\n")
    return kg_v4


if __name__ == "__main__":
    prepare_data_v4()
