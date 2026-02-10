import json
import config
import asyncio
from neo4j import GraphDatabase
from pydantic import validate_call
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, KGWriterModel

class Neo4jCreateWriter(KGWriter):
    """관계에 대해 MERGE 대신 CREATE를 사용하는 Custom KGWriter (에피소드별로 다른 관계도 반영)"""

    def __init__(self, driver, neo4j_database=None):
        self.driver = driver
        self.neo4j_database = neo4j_database


    def _wipe_database(self) -> None:
        self.driver.execute_query(
            "MATCH (n) DETACH DELETE n",
            database_=self.neo4j_database,
        )

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        try:
            self._wipe_database()
            with self.driver.session(database=self.neo4j_database) as session:
                # 1. node 작성
                for node in graph.nodes:
                    if not node.label or not node.label.strip():
                        print(f"Skipping node with empty label: {node}")
                        continue
                    # Clean label
                    clean_label = node.label.strip().replace(" ", "_").replace(",", "") 
                    labels = f":`{clean_label}`"
                    session.run(
                        f"""
                        MERGE (n{labels} {{id: $id}})
                        SET n += $props
                        """,
                        {"id": node.id, "props": node.properties or {}},
                    )

                # 2. relationship 작성
                for rel in graph.relationships:
                    if not rel.type or not rel.type.strip():
                        print(f"Skipping relationship with empty type: {rel}")
                        continue
                    # Clean type
                    clean_type = rel.type.strip().replace(" ", "_")
                    session.run(
                        f"""
                        MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
                        CREATE (a)-[r:`{clean_type}`]->(b)
                        SET r += $props
                        """,
                        {
                            "start_id": rel.start_node_id,
                            "end_id": rel.end_node_id,
                            "props": rel.properties or {},
                        },
                    )

            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "node_count": len(graph.nodes),
                    "relationship_count": len(graph.relationships),
                },
            )
        except Exception as e:
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})

async def write_to_neo4j(graph: Neo4jGraph):
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    
    # writer = KGWriter(driver)
    writer = Neo4jCreateWriter(driver, neo4j_database=config.NEO4J_DATABASE)
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
