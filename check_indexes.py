from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "admin123")
DATABASE = os.getenv("NEO4J_DATABASE", "db_graphrag_agent")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def check_indexes():
    with driver.session(database=DATABASE) as session:
        result = session.run("SHOW VECTOR INDEXES")
        print(f"{'Name':<40} | {'Entity':<20} | {'Labels/Types':<20}")
        print("-" * 90)
        for record in result:
             # Depending on Neo4j version, fields might differ. 
             # Commonly 'name', 'entityType', 'labelsOrTypes'
             print(f"{record.get('name', 'N/A'):<40} | {record.get('entityType', 'N/A'):<20} | {str(record.get('labelsOrTypes', 'N/A')):<20}")

if __name__ == "__main__":
    check_indexes()
