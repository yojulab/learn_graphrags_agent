"""
[Apache AGE Track Common] PostgreSQL 접속, AGE 로드, agtype 파서 및 SQL 헬퍼
(src/03_apache_age_track/common_age.py)
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import openai

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import config

def get_age_connection():
    """PostgreSQL 연결 및 AGE 확장 로드"""
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=config.POSTGRES_DB
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("LOAD 'age';")
        cur.execute('SET search_path = ag_catalog, "$user", public;')
    return conn

def escape_cypher_str(val: Any) -> str:
    """Cypher 쿼리 내 문자열 리터럴 안전 이스케이프"""
    if val is None:
        return ""
    s = str(val)
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", " ")

def parse_agtype(val: Any) -> Any:
    """Apache AGE agtype 파싱 보조 함수"""
    if val is None:
        return None
    s = str(val)
    # agtype 숫자 접미사(::numeric, ::vertex 등) 정제
    s = re.sub(r'::\w+', '', s)
    return s

class AGEEmbedder:
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
                print(f"⚠️ 임베딩 실패: {e}")
                all_embeddings.extend([[0.0] * config.EMBEDDING_DIMENSION for _ in batch])
        return all_embeddings
