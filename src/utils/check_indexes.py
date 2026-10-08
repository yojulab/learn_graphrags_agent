import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase
import config

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def check_indexes():
    with driver.session(database=config.NEO4J_DATABASE) as session:
        result = session.run("SHOW VECTOR INDEXES")
        print(f"{'Name':<40} | {'Entity':<20} | {'Labels/Types':<20}")
        print("-" * 90)
        for record in result:
             print(f"{record.get('name', 'N/A'):<40} | {record.get('entityType', 'N/A'):<20} | {str(record.get('labelsOrTypes', 'N/A')):<20}")

if __name__ == "__main__":
    check_indexes()
