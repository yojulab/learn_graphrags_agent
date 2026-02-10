import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database (Neo4j)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "db-demonslayer")

# AI / LLM
LLM_MODEL = os.getenv("LLM_MODEL", "ingu627/exaone4.0:1.2b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# Vector Search Configuration
VECTOR_INDEX_NODE = "entity_embeddings"
VECTOR_INDEX_RELATIONSHIP = "relationship_embeddings"
EMBEDDING_DIMENSION = 1024  # bge-m3 모델 차원
