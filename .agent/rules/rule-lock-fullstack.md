---
trigger: always_on
---

# Agent Harness Rules & Architecture Lock

## 1. MCP Server & Tool Usage Guidelines

* **context7**:
  * Defines the current working scope only.
  * Never assume global project context beyond what is provided.
  * Scope: Working on specific scripts, modules, or data processing pipelines.
  * Prohibition: No arbitrary refactoring or touching unrelated files without explicit task requirement.
* **sequential-thinking**:
  * Mandatory for breaking down complex problems, planning multi-step workflows, and maintaining context.
  * Use for data pipeline design, knowledge graph schema planning, and agent workflow implementation.

---

## 2. Project Overview & Architecture

**Project Type:** GraphRAG (Graph Retrieval-Augmented Generation) Agent  
**Target Directory:** `/apps/learn_graphrags_agent`

### Purpose
Build a knowledge graph from anime plot summaries (Demon Slayer Season 1) and enable natural language querying through an AI agent that converts questions to Cypher queries and performs hybrid retrieval (vector similarity + graph traversal).

### Core Components & Folder Hierarchy
```text
/apps/learn_graphrags_agent
├── .env                       # Centralized Environment Variables
├── config.py                  # Environment Configuration Loader
├── PRD.md                     # Product Requirements Document & Project Analysis
├── README.md                  # Project Quickstart & Usage Instructions
├── dockers/                   # Infrastructure (Neo4j, Postgres AGE, Mongo, Ollama, App)
│   ├── .env
│   ├── docker-compose.yml
│   └── Dockerfile.*
├── output/                    # Pipeline Output JSON Artifacts
├── src/                       # Executable Pipeline Scripts by Version
│   ├── app/                   # GraphRAG Web Application & Dashboard (FastAPI)
│   │   ├── __init__.py
│   │   └── main.py
│   ├── v1/                    # Basic Version (OpenAI Schema Baseline)
│   │   ├── 1_prepare_data_v1.py
│   │   ├── 2_ingest_data_v1.py
│   │   └── 3_graphrag_agent_v1.py
│   ├── v2/                    # Intermediate Version (Metadata Filtering + Reranking)
│   │   ├── 1_prepare_data_v2.py
│   │   ├── 2_ingest_data_v2.py
│   │   └── 3_graphrag_agent_v2.py
│   ├── v3/                    # Production Hybrid Version (Vector + Graph Expansion + Text2Cypher)
│   │   ├── 1_prepare_data_v3.py
│   │   ├── 2_ingest_data_v3.py
│   │   └── 3_graphrag_agent_v3.py
│   └── utils/
│       ├── check_indexes.py
│       ├── playwright_scraper.py
│       └── test_pipeline_playwright.py
└── docs/
    └── agent/
        ├── harness_prompt.md  # Agent System Harness Prompt Standard
        └── system_prompt.md   # Agent System Prompt Standard
```

---

## 3. Configuration & Environment Management (.env Lock)

* All runtime configurations must be driven by `.env` (root directory or `dockers/.env`).
* `config.py` acts as the single source of truth python interface to `.env`.
* Configurable parameters:
  * **Neo4j**: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
  * **LLM & Embeddings**: `LLM_MODEL`, `EMBEDDING_MODEL`, `MODEL_API_URL`, `OPENAI_API_KEY`
  * **Retrieval & Graph Search**: `USE_HYBRID_RETRIEVAL`, `VECTOR_TOP_K`, `GRAPH_EXPANSION_DEPTH`
  * **Vector Indexes**: `VECTOR_INDEX_NODE`, `VECTOR_INDEX_RELATIONSHIP_PREFIX`, `EMBEDDING_DIMENSION`
  * **Master Schemas**: `NODE_LABELS`, `RELATIONSHIP_TYPES`

---

## 4. Development Rules & Code Standards

### Type Safety & Python Standard
* **Python Version**: Python 3.12+ with strict typing annotations.
* **Validation**: Use Pydantic models for data parsing and validation.
* **Path Resolution**: All versioned sub-scripts in `src/v*/` must include root path resolution to import `config.py` seamlessly.

### Data Pipeline Rules
* **Preparation (`1_prepare_data_v*.py`)**: Scrape source content, validate entity/relationship JSON, save to `output/`.
* **Ingestion (`2_ingest_data_v*.py`)**: Create Neo4j nodes/relationships with vector embeddings, build vector indexes (`entity_embeddings_*`, `rel_embeddings_*`).
* **Agent Execution (`3_graphrag_agent_v*.py`)**: Hybrid vector retrieval → Graph expansion → Text2Cypher generation → Answer synthesis.

---

## 5. Harness Prompt Specifications

All LLM calls within the agent pipeline must adhere to standardized system prompts:
1. **Schema Context**: Always supply valid node labels, relationship types, and few-shot Cypher templates.
2. **Thinking Tag Sanitization**: Clean model outputs containing `<think>...</think>` tags (e.g. EXAONE thinking models).
3. **Cypher Safety**: Enforce read-only Cypher generation, parametric matching, and syntax checking before execution.