import json
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
    """Note embedding text generation"""
    return f"{node.label}: {node.properties.get('name', node.id)}"

def create_relationship_text(rel: Neo4jRelationship, node_map: Dict[str, Neo4jNode]) -> str:
    """Relationship embedding text generation"""
    start_name = node_map[rel.start_node_id].properties.get('name', rel.start_node_id)
    end_name = node_map[rel.end_node_id].properties.get('name', rel.end_node_id)
    
    # Priority: description > context > action/technique
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
    """OpenAI API compatible embeddings (Supports Ollama)"""
    
    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """Single text embedding"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch text embedding (Max 100)"""
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
                # Fallback to zero embedding
                all_embeddings.extend([[0.0] * config.EMBEDDING_DIMENSION for _ in batch])
        
        return all_embeddings


class Neo4jCreateWriter(KGWriter):
    """Custom KGWriter to use CREATE instead of MERGE for relationships (to handle multiple episodes)"""

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
        """Create Node and Relationship Vector Indexes using Config"""
        
        # 1. Node Vector Indexes
        for label in config.NODE_LABELS:
            try:
                index_name = f"{config.VECTOR_INDEX_NODE}_{label}"
                self.driver.execute_query(
                    f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label})
                    ON n.embedding
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {config.EMBEDDING_DIMENSION},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                    """,
                    database_=self.neo4j_database,
                )
                print(f"✅ Vector index '{index_name}' created")
            except Exception as e:
                print(f"⚠️  Failed to create entity vector index for {label}: {e}")
        
        # 2. Relationship Vector Indexes (Neo4j 5.13+)
        # Using the centralized RELATIONSHIP_TYPES from config
        try:
            for rel_type in config.RELATIONSHIP_TYPES:
                try:
                    index_name = f"{config.VECTOR_INDEX_RELATIONSHIP_PREFIX}_{rel_type}"
                    self.driver.execute_query(
                        f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR ()-[r:{rel_type}]-()
                        ON r.embedding
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {config.EMBEDDING_DIMENSION},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """,
                        database_=self.neo4j_database,
                    )
                    print(f"✅ Vector index '{index_name}' created")
                except Exception as e:
                    print(f"⚠️  Failed to create relationship vector index for {rel_type}: {e}")
        except Exception as e:
            print(f"ℹ️  Relationship vector indexes check failed: {e}")

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        try:
            start_time = time.time()
            self._wipe_database()
            
            # Create Vector Indexes
            if self.embedder:
                print("\n🔍 Creating vector indexes...")
                self._create_vector_indexes()
            
            with self.driver.session(database=self.neo4j_database) as session:
                # 1. Generate Node Embeddings
                node_embeddings = []
                if self.embedder:
                    print(f"\n🧠 Generating embeddings for {len(graph.nodes)} nodes...")
                    node_texts = [create_node_text(node) for node in graph.nodes]
                    node_embeddings = self.embedder.embed_documents(node_texts)
                    print(f"✅ Generated {len(node_embeddings)} node embeddings")
                
                # 2. Save Nodes + Embeddings
                for i, node in enumerate(tqdm(graph.nodes, desc="Creating nodes", unit="node")):
                    if not node.label or not node.label.strip():
                        print(f"Skipping node with empty label: {node}")
                        continue
                    
                    # Clean label
                    clean_label = node.label.strip().replace(" ", "_").replace(",", "") 
                    labels = f":`{clean_label}`"
                    
                    # Add Embedding
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

                # 3. Generate Relationship Embeddings
                rel_embeddings = []
                if self.embedder:
                    print(f"\n🧠 Generating embeddings for {len(graph.relationships)} relationships...")
                    node_map = {node.id: node for node in graph.nodes}
                    rel_texts = [create_relationship_text(rel, node_map) for rel in graph.relationships]
                    rel_embeddings = self.embedder.embed_documents(rel_texts)
                    print(f"✅ Generated {len(rel_embeddings)} relationship embeddings")

                # 4. Save Relationships + Embeddings
                for i, rel in enumerate(tqdm(graph.relationships, desc="Creating relationships", unit="rel")):
                    if not rel.type or not rel.type.strip():
                        print(f"Skipping relationship with empty type: {rel}")
                        continue
                    
                    # Clean type
                    clean_type = rel.type.strip().replace(" ", "_")
                    
                    # Add Embedding
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
    driver = GraphDatabase.driver(
        config.NEO4J_URI, 
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
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
    # Load same data as v2/v3
    with open("output/knowledge_graph_v3.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = [Neo4jNode(**node) for node in data["nodes"]]
    relationships = [Neo4jRelationship(**rel) for rel in data.get("relationships", []) if rel.get("type")]
    graph = Neo4jGraph(nodes=nodes, relationships=relationships)

    asyncio.run(write_to_neo4j(graph))
