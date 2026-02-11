import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database (Neo4j)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://db_neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "db_graphrag_agent")

# ============================================================
# Hybrid Retrieval Settings
# ============================================================
USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))
GRAPH_EXPANSION_DEPTH = int(os.getenv("GRAPH_EXPANSION_DEPTH", "2"))


# AI / LLM
# LLM_MODEL = os.getenv("LLM_MODEL", "sam860/exaone-4.0:1.2b-Q8_0")
LLM_MODEL = os.getenv("LLM_MODEL", "sam860/exaone-4.0:1.2b-thinking-Q8_0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://ollama:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# Vector Search Configuration
VECTOR_INDEX_NODE = "entity_embeddings" # Basic Node Index (if used)
VECTOR_INDEX_RELATIONSHIP_PREFIX = "rel_embeddings"
EMBEDDING_DIMENSION = 1024  # bge-m3 model dimension

# Centralized Relationship Types (English) - Used for Indexing & Querying
RELATIONSHIP_TYPES = [
    "FIGHTS", "PROTECTS", "TRAINS", "TRAINS_WITH", "KNOWS", 
    "FAMILY_OF", "SIBLING_OF", "ALLY_OF", "ENEMY_OF", 
    "DEFEATS", "SAVES", "RESCUES", "MEETS", "ENCOUNTERS", 
    "GUIDES", "ATTACKS", "DEFENDS", "TRANSFORMS", 
    "JOINS", "SUPPORTS", "REUNITES_WITH", "BATTLES", 
    "HEALS", "TEACHES"
]

# Valid Node Labels
NODE_LABELS = ["인간", "도깨비"]
