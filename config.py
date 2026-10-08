import os
from pathlib import Path
from dotenv import load_dotenv

# Find project root directory and load .env files
BASE_DIR = Path(__file__).resolve().parent
ROOT_ENV = BASE_DIR / ".env"
DOCKER_ENV = BASE_DIR / "dockers" / ".env"

if ROOT_ENV.exists():
    load_dotenv(dotenv_path=ROOT_ENV)
elif DOCKER_ENV.exists():
    load_dotenv(dotenv_path=DOCKER_ENV)
else:
    load_dotenv()

# ============================================================
# Project & Infrastructure Settings
# ============================================================
PROJECT_NAME = os.getenv("PROJECT_NAME", "graphrag_agent")
COMPOSE_PROFILES = os.getenv("COMPOSE_PROFILES", "neo4j,postgres,mongo,ollama,app")

# Database (Neo4j)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "db-graphrag-agent")

# Database (PostgreSQL / AGE)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "db_graphrag_agent")

# Database (MongoDB)
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "admin123")
MONGO_DB = os.getenv("MONGO_DB", "db_graphrag_agent")

# ============================================================
# AI / LLM & Embedding Settings
# ============================================================
LLM_MODEL = os.getenv("LLM_MODEL", "sam860/exaone-4.0:1.2b-thinking-Q8_0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# ============================================================
# Hybrid Retrieval & GraphRAG Settings
# ============================================================
USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() in ("true", "1", "t", "yes")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))
GRAPH_EXPANSION_DEPTH = int(os.getenv("GRAPH_EXPANSION_DEPTH", "2"))

# Vector Search Configuration
VECTOR_INDEX_NODE = os.getenv("VECTOR_INDEX_NODE", "entity_embeddings")
VECTOR_INDEX_RELATIONSHIP_PREFIX = os.getenv("VECTOR_INDEX_RELATIONSHIP_PREFIX", "rel_embeddings")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# Centralized Relationship Types
_default_rel_types = (
    "FIGHTS,PROTECTS,TRAINS,TRAINS_WITH,KNOWS,FAMILY_OF,SIBLING_OF,ALLY_OF,ENEMY_OF,"
    "DEFEATS,SAVES,RESCUES,MEETS,ENCOUNTERS,GUIDES,ATTACKS,DEFENDS,TRANSFORMS,"
    "JOINS,SUPPORTS,REUNITES_WITH,BATTLES,HEALS,TEACHES"
)
raw_rel_types = os.getenv("RELATIONSHIP_TYPES", _default_rel_types)
RELATIONSHIP_TYPES = [r.strip() for r in raw_rel_types.split(",") if r.strip()]

# Valid Node Labels
raw_node_labels = os.getenv("NODE_LABELS", "인간,도깨비")
NODE_LABELS = [lbl.strip() for lbl in raw_node_labels.split(",") if lbl.strip()]
