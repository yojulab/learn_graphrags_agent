import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
import asyncio
import time
import openai
from typing import List, Dict
from neo4j import GraphDatabase
from pydantic import validate_call
from tqdm import tqdm
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, KGWriterModel

# ============================================================
# Embedding Helper Functions
# ============================================================

def create_node_text(node: Neo4jNode) -> str:
    """노드 임베딩용 텍스트 생성"""
    return f"{node.label}: {node.properties.get('name', node.id)}"

def create_relationship_text(rel: Neo4jRelationship, node_map: Dict[str, Neo4jNode]) -> str:
    """관계 임베딩용 텍스트 생성"""
    start_name = node_map[rel.start_node_id].properties.get('name', rel.start_node_id)
    end_name = node_map[rel.end_node_id].properties.get('name', rel.end_node_id)
    
    # description 우선, 없으면 context, action, technique 조합
    if not rel.properties:
        content = ""
    else:
        content = rel.properties.get('description') or \
                  rel.properties.get('context') or \
                  f"{rel.properties.get('action', '')} {rel.properties.get('technique', '')}".strip()
    
    return f"{rel.type} - {start_name} → {end_name}: {content}"

# ============================================================
# OpenAI-compatible Embeddings Class
# ============================================================

class OpenAIEmbeddings:
    """OpenAI API 호환 임베딩 생성 (Ollama 지원)"""
    
    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """배치 텍스트 임베딩 (최대 100개씩)"""
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                all_embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"⚠️  Embedding batch {i//batch_size + 1} failed: {e}")
                # 실패한 배치는 빈 임베딩으로 대체
                all_embeddings.extend([[0.0] * 1024 for _ in batch])
        
        return all_embeddings


class Neo4jCreateWriter(KGWriter):
    """관계에 대해 MERGE 대신 CREATE를 사용하는 Custom KGWriter (에피소드별로 다른 관계도 반영)"""

    def __init__(self, driver, neo4j_database=None, embedder=None):
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.embedder = embedder


    def _wipe_database(self) -> None:
        self.driver.execute_query(
            "MATCH (n) DETACH DELETE n",
            database_=self.neo4j_database,
        )
    
    def _create_vector_indexes(self) -> None:
        """노드 및 관계 벡터 인덱스 생성"""
        # 노드 벡터 인덱스 (각 라벨마다 별도 인덱스 생성)
        entity_labels = ["인간", "도깨비"]
        for label in entity_labels:
            try:
                index_name = f"entity_embeddings_{label}"
                self.driver.execute_query(
                    f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label})
                    ON n.embedding
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }}}}
                    """,
                    database_=self.neo4j_database,
                )
                print(f"✅ Vector index '{index_name}' created")
            except Exception as e:
                print(f"⚠️  Failed to create entity vector index for {label}: {e}")
        
        # 관계 벡터 인덱스 (Neo4j 5.13+ only, optional)
        # Note: 관계 인덱스는 Neo4j 5.13+에서만 지원됨
        # 현재 버전에서 지원하지 않으면 스킵
        try:
            # 관계 타입별 인덱스 생성 (Neo4j 5.13+에서는 관계 타입 지정 필요)
            relationship_types = ["전투", "변화", "사용", "상호작용", "소속", "보유", "훈련"]
            for rel_type in relationship_types:
                try:
                    index_name = f"rel_embeddings_{rel_type}"
                    self.driver.execute_query(
                        f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR ()-[r:{rel_type}]-()
                        ON r.embedding
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """,
                        database_=self.neo4j_database,
                    )
                    print(f"✅ Vector index '{index_name}' created")
                except Exception as e:
                    # 관계 벡터 인덱스는 선택 사항이므로 에러 무시
                    pass
        except Exception as e:
            print(f"ℹ️  Relationship vector indexes not created (requires Neo4j 5.13+): {e}")

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        try:
            start_time = time.time()
            self._wipe_database()
            
            # 벡터 인덱스 생성
            if self.embedder:
                print("\n🔍 Creating vector indexes...")
                self._create_vector_indexes()
            
            with self.driver.session(database=self.neo4j_database) as session:
                # 1. 노드 임베딩 생성
                node_embeddings = []
                if self.embedder:
                    print(f"\n🧠 Generating embeddings for {len(graph.nodes)} nodes...")
                    node_texts = [create_node_text(node) for node in graph.nodes]
                    node_embeddings = self.embedder.embed_documents(node_texts)
                    print(f"✅ Generated {len(node_embeddings)} node embeddings")
                
                # 2. 노드 + 임베딩 저장
                for i, node in enumerate(tqdm(graph.nodes, desc="Creating nodes", unit="node")):
                    if not node.label or not node.label.strip():
                        print(f"Skipping node with empty label: {node}")
                        continue
                    
                    # Clean label
                    clean_label = node.label.strip().replace(" ", "_").replace(",", "") 
                    labels = f":`{clean_label}`"
                    
                    # 임베딩 추가
                    query_params = {"id": node.id, "props": node.properties or {}}
                    if self.embedder and i < len(node_embeddings):
                        query = f"""
                        MERGE (n{labels} {{id: $id}})
                        SET n += $props, n.embedding = $embedding
                        """
                        query_params["embedding"] = node_embeddings[i]
                    else:
                        query = f"""
                        MERGE (n{labels} {{id: $id}})
                        SET n += $props
                        """
                    
                    session.run(query, query_params)

                # 3. 관계 임베딩 생성
                rel_embeddings = []
                if self.embedder:
                    print(f"\n🧠 Generating embeddings for {len(graph.relationships)} relationships...")
                    node_map = {node.id: node for node in graph.nodes}
                    rel_texts = [create_relationship_text(rel, node_map) for rel in graph.relationships]
                    rel_embeddings = self.embedder.embed_documents(rel_texts)
                    print(f"✅ Generated {len(rel_embeddings)} relationship embeddings")

                # 4. 관계 + 임베딩 저장
                for i, rel in enumerate(tqdm(graph.relationships, desc="Creating relationships", unit="rel")):
                    if not rel.type or not rel.type.strip():
                        print(f"Skipping relationship with empty type: {rel}")
                        continue
                    
                    # Clean type
                    clean_type = rel.type.strip().replace(" ", "_")
                    
                    # 임베딩 추가
                    query_params = {
                        "start_id": rel.start_node_id,
                        "end_id": rel.end_node_id,
                        "props": rel.properties or {},
                    }
                    
                    if self.embedder and i < len(rel_embeddings):
                        query = f"""
                        MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
                        CREATE (a)-[r:`{clean_type}`]->(b)
                        SET r += $props, r.embedding = $embedding
                        """
                        query_params["embedding"] = rel_embeddings[i]
                    else:
                        query = f"""
                        MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
                        CREATE (a)-[r:`{clean_type}`]->(b)
                        SET r += $props
                        """
                    
                    session.run(query, query_params)

            elapsed_time = time.time() - start_time
            print(f"\n⏱️  Graph ingestion completed in {elapsed_time:.2f}s")
            
            metadata = {
                "node_count": len(graph.nodes),
                "relationship_count": len(graph.relationships),
                "execution_time": f"{elapsed_time:.2f}s"
            }
            if self.embedder:
                metadata["node_embeddings_count"] = len(node_embeddings)
                metadata["relationship_embeddings_count"] = len(rel_embeddings)
            
            return KGWriterModel(status="SUCCESS", metadata=metadata)
        except Exception as e:
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})

async def write_to_neo4j(graph: Neo4jGraph):
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    
    # Initialize embedder
    embedder = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.MODEL_API_URL
    )
    
    writer = Neo4jCreateWriter(driver, neo4j_database=config.NEO4J_DATABASE, embedder=embedder)
    result = await writer.run(graph)
    print(result)


if __name__ == "__main__":
    # 검증된 데이터 사용 (1_prepare_data_v3.py에서 생성)
    with open("output/knowledge_graph_v3.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = [Neo4jNode(**node) for node in data["nodes"]]
    relationships = [Neo4jRelationship(**rel) for rel in data.get("relationships", []) if rel.get("type")]
    graph = Neo4jGraph(nodes=nodes, relationships=relationships)

    asyncio.run(write_to_neo4j(graph))
