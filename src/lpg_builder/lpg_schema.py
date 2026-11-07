"""
[LPG Schema] Labeled Property Graph (LPG) 표준 온톨로지 및 Pydantic v2 데이터 모델
(src/01_lpg_builder/2_lpg_schema.py)

LPG 3대 핵심 구성요소:
1. Node (Vertex): 레이블(Label: :인간, :도깨비) + 고유 식별자(id) + 속성(Properties)
2. Edge (Relationship): 방향성(start_id -> end_id) + 단일 관계 유형(Type: 24종) + 속성(Properties)
3. Property (Key-Value Map): 정점/간선에 부가되는 상세 메타데이터 (에피소드, 전투/지원 속성, 임베딩 등)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

# ============================================================
# 1. LPG 마스터 엔티티 정의 (Standard Entity Dictionary)
# ============================================================

VALID_NODES: Dict[str, Dict[str, str]] = {
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

# 24개 표준 관계 유형 (config.RELATIONSHIP_TYPES와 동기화)
VALID_RELATIONSHIP_TYPES: List[str] = config.RELATIONSHIP_TYPES

# 영문/별칭 ➔ 한국어 표준 이름 매핑
ENGLISH_TO_KOREAN_NAME: Dict[str, str] = {
    "Tanjiro Kamado": "카마도 탄지로", "Tanjiro": "카마도 탄지로",
    "Nezuko Kamado": "카마도 네즈코", "Nezuko": "카마도 네즈코",
    "Giyu Tomioka": "토미오카 기유", "Giyu": "토미오카 기유",
    "Sakonji Urokodaki": "우로코다키 사콘지", "Sakonji": "우로코다키 사콘지",
    "Sabito": "사비토",
    "Makomo": "마코모",
    "Zenitsu Agatsuma": "아가츠마 젠이츠", "Zenitsu": "아가츠마 젠이츠",
    "Inosuke Hashibira": "하시비라 이노스케", "Inosuke": "하시비라 이노스케",
    "Kanao Tsuyuri": "츠유리 카나오", "Kanao": "츠유리 카나오",
    "Kyojuro Rengoku": "렌고쿠 쿄쥬로", "Rengoku": "렌고쿠 쿄쥬로",
    "Kagaya Ubuyashiki": "우부야시키 카가야",
    "Shinobu Kocho": "코쵸우 시노부", "Shinobu": "코쵸우 시노부",
    "Sanemi Shinazugawa": "시나즈가와 사네미", "Sanemi": "시나즈가와 사네미",
    "Muzan Kibutsuji": "키부츠지 무잔", "Muzan": "키부츠지 무잔",
    "Susamaru": "스사마루",
    "Yahaba": "야하바",
    "Kyogai": "쿄우가이",
    "Rui": "루이",
    "Enmu": "엔무",
}

NAME_TO_ID: Dict[str, str] = {v["name"]: k for k, v in VALID_NODES.items()}

# ============================================================
# 2. Pydantic v2 LPG 데이터 모델
# ============================================================

class LPGNode(BaseModel):
    """
    LPG 정점(Node/Vertex) 모델
    - label: 노드 레이블 (:인간 또는 :도깨비)
    - id: 고유 식별자 (예: N0, N1)
    - name: 표준 한국어 인명
    - properties: 임베딩 및 메타데이터를 포함하는 Key-Value 속성 맵
    """
    id: str
    label: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        clean = v.strip()
        if clean not in config.NODE_LABELS:
            raise ValueError(f"유효하지 않은 LPG 정점 레이블: '{clean}'. 허용 목록: {config.NODE_LABELS}")
        return clean

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("LPG 정점 name은 필수값입니다.")
        return v.strip()


class LPGEdge(BaseModel):
    """
    LPG 간선(Edge/Relationship) 모델
    - type: 24개 표준 관계 동사 (대문자 스네이크 케이스)
    - start_id: 시작 정점 ID
    - end_id: 도착 정점 ID
    - properties: 에피소드 정보, 전투/지원 결과, 문맥, 임베딩을 담는 속성 맵
    """
    type: str
    start_id: str
    end_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in config.RELATIONSHIP_TYPES:
            raise ValueError(f"유효하지 않은 LPG 관계 타입: '{clean}'. 허용 목록: {config.RELATIONSHIP_TYPES}")
        return clean


class LPGGraph(BaseModel):
    """
    LPG 지식 그래프 통합 컨테이너 모델
    - version: LPG 스키마 버전
    - nodes: LPG 정점 목록
    - edges: LPG 간선 목록
    - metadata: 추출 일시, 소스 통계 등
    """
    version: str = "lpg-v1.0"
    nodes: List[LPGNode]
    edges: List[LPGEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)
