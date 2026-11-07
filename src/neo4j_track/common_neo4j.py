"""
[Neo4j Track Common] Neo4j 접속, 드라이버, Think 태그 정제 및 공통 임베딩 유틸
(src/02_neo4j_track/common_neo4j.py)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import openai
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.llm.types import LLMResponse

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

def get_neo4j_driver():
    """싱글톤 Neo4j 드라이버 인스턴스 반환"""
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )

class CleanOpenAILLM(OpenAILLM):
    """
    추론 모델(sam860/exaone-4.0 등)의 <think>...</think> 블록을
    정규식으로 사전 제거하여 안전한 Cypher/응답을 보장하는 커스텀 LLM
    """
    def invoke(self, input_text: str) -> LLMResponse:
        response = super().invoke(input_text)
        content = response.content

        # 1. <think>...</think> 블록 제거
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.replace('</think>', '').strip()

        # 2. 마크다운 코드 블록(```cypher ... ```) 추출
        code_block_match = re.search(r'```(?:cypher)?(.*?)```', content, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            content = code_block_match.group(1).strip()

        response.content = content
        return response

class Neo4jEmbedder:
    """1024차원 bge-m3 임베딩 생성기"""
    def __init__(self, model: str = config.EMBEDDING_MODEL):
        self.client = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.MODEL_API_URL,
            timeout=300.0
        )
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        res = self.client.embeddings.create(model=self.model, input=[text])
        return res.data[0].embedding

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                res = self.client.embeddings.create(model=self.model, input=batch)
                all_embeddings.extend([d.embedding for d in res.data])
            except Exception as e:
                print(f"⚠️ 임베딩 배치 오류: {e}")
                all_embeddings.extend([[0.0] * config.EMBEDDING_DIMENSION for _ in batch])
        return all_embeddings

SCHEMA_NODES = """
## 노드 타입 (Node Labels):
- :인간 {id: STRING, name: STRING, embedding: LIST[FLOAT]}
- :도깨비 {id: STRING, name: STRING, embedding: LIST[FLOAT]}
"""

SCHEMA_RELATIONSHIPS = f"""
## 관계 타입 (Relationship Types - 24종):
{', '.join(config.RELATIONSHIP_TYPES)}
- 공통 속성: episode_number (STRING), season (INT), episode (INT), context (STRING), description (STRING)
- 전투/행동 속성: outcome, action, technique, enemy
- 지원/역할 속성: role, method, duration, to, subject
"""
