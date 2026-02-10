---
trigger: always_on
---

## 1. MCP Server Usage Guidelines

The following MCP servers must be utilized as described:

*   **context7**:
    *   Defines the current working scope only.
    *   Never assume global project context beyond what is provided.
    *   Allowed: Working on specific scripts, modules, or data processing pipelines.
    *   Forbidden: Global redesigns or touching unrelated files without explicit approval.
*   **sequential-thinking**:
    *   Mandatory for breaking down complex problems, planning multi-step workflows, and maintaining context.
    *   Use for data pipeline design, knowledge graph schema planning, and agent workflow implementation.

---

## 2. Project Overview & Architecture

**Project Type:** GraphRAG (Graph Retrieval-Augmented Generation) Agent  
**Target Directory:** `/Users/sanghunoh/Develops/repository/lectues/learn_graphrags_agent`

### Purpose
Build a knowledge graph from anime plot summaries (Demon Slayer Season 1) and enable natural language querying through an AI agent that converts questions to Cypher queries.

### Core Components
1. **Data Preparation** (`1_prepare_data_v*.py`) - Wikipedia scraping & entity extraction
2. **Data Ingestion** (`2_ingest_data.py`) - Neo4j knowledge graph creation
3. **GraphRAG Agent** (`3_graphrag_agent.py`) - Natural language to Cypher query conversion
4. **Configuration** (`config.py`) - Centralized environment variables

---

## 3. Technology Stack (Non-Negotiable)

### Core Technologies
*   **Python:** 3.12+ (strict type hints required)
*   **Graph Database:** Neo4j 5+ (via Docker)
*   **LLM Provider:** OpenAI API or Ollama (local)
*   **AI Framework:** LangChain / LangGraph (for agent orchestration)
*   **Graph Library:** `neo4j-graphrag` (official Neo4j GraphRAG library)

### Architecture Lock
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Knowledge Graph | Neo4j | Graph storage & Cypher querying |
| LLM | OpenAI / Ollama | Entity extraction & natural language processing |
| Orchestration | LangChain/LangGraph | Agent workflow & state management |
| Data Processing | BeautifulSoup, Requests | Web scraping & data extraction |
| Config | python-dotenv | Environment variable management |

---

## 4. Development Rules & Best Practices

### 1. Type Safety & Code Quality
*   **Strict Typing:** All functions must use Python 3.12+ type hints
*   **No `Any`:** Use Pydantic models or specific types
*   **Validation:** Use Pydantic for data validation in JSON processing
*   **Documentation:** Docstrings required for all public functions
*   **Error Handling:** Explicit try-except blocks with meaningful error messages

### 2. Data Pipeline Rules

#### Data Preparation (`1_prepare_data_v*.py`)
*   **Extraction Schema:** Define clear entity and relationship types upfront
*   **Prompt Engineering:** Store LLM prompts in dedicated variables/files, not hardcoded
*   **Data Validation:** Validate JSON structure before saving to `output/`
*   **Caching:** Implement cache mode to avoid redundant API calls
*   **Output Format:** Standardized JSON with `nodes` and `relationships` keys

#### Data Ingestion (`2_ingest_data.py`)
*   **Idempotency:** Clear existing data before ingestion to prevent duplicates
*   **Atomic Operations:** Use Neo4j transactions for batch operations
*   **Relationship Creation:** Use `CREATE` or `MERGE` consistently based on schema
*   **Error Recovery:** Log failed node/relationship creation, continue processing

#### GraphRAG Agent (`3_graphrag_agent.py`)
*   **Prompt Clarity:** Cypher generation prompts must include schema context
*   **Query Validation:** Validate generated Cypher before execution
*   **Result Processing:** Always provide context from graph in final answer
*   **Hallucination Prevention:** Never generate answers without Neo4j query results

### 3. Neo4j & Cypher Best Practices
*   **Schema Enforcement:** Define node labels and relationship types in config
*   **Index Usage:** Create indexes on frequently queried properties
*   **Query Optimization:** Use `LIMIT` clauses to prevent performance issues
*   **Parameterization:** Use parameterized Cypher queries to prevent injection
*   **Relationship Direction:** Respect semantic direction in relationships

### 4. LangChain/LangGraph Implementation

#### State Management
*   **Typed States:** Define Pydantic models for agent state
*   **Node Granularity:** One specific task per node (e.g., extract_query, generate_cypher, execute_query, format_answer)
*   **Edge Conditions:** Explicit conditional edges based on state validation

#### Graph Structure
```python
# Example structure
StateGraph → [query_analyzer → cypher_generator → query_executor → answer_formatter]
```

#### Model Configuration
*   **OpenAI:** Use `gpt-4o-mini` or `gpt-4` based on complexity
*   **Ollama:** Use local models from `config.py` (e.g., `ingu627/exaone4.0:1.2b`)
*   **Fallback:** Always define fallback models in case primary fails

### 5. Configuration Management
*   **Environment Variables:** All secrets and configs in `.env` (never commit)
*   **Config File:** Centralized in `config.py` with sensible defaults
*   **Neo4j Connection:** Use environment variables for URI, user, password, database
*   **API Keys:** Load from `.env` with `python-dotenv`

---

## 5. File Structure & Discipline

### Mandatory File Naming
*   **Pipeline Scripts:** `{step}_{description}_v{version}.py` (e.g., `1_prepare_data_v2.py`)
*   **Config:** Single `config.py` at project root
*   **Output:** JSON files in `output/` directory with descriptive Korean names
*   **Docker:** Infrastructure in `dockers/` directory

### File Responsibilities
| File | Responsibility | Dependencies |
|------|---------------|--------------|
| `1_prepare_data_v*.py` | Wikipedia scraping, entity extraction via LLM | BeautifulSoup, OpenAI, Requests |
| `2_ingest_data.py` | Neo4j data ingestion from JSON | neo4j-graphrag, Neo4j driver |
| `3_graphrag_agent_v*.py` | Natural language to Cypher conversion | LangChain, OpenAI, Neo4j |
| `config.py` | Environment variable loading | python-dotenv |

---

## 6. Sequential Thinking Enforcement

For any complex task, follow this sequence:

### Data Pipeline Development
1. **Schema Design** - Define nodes, relationships, properties
2. **Prompt Engineering** - Create LLM extraction prompts
3. **Script Implementation** - Write extraction/ingestion logic
4. **Validation** - Test with sample data
5. **Verification** - Check Neo4j data quality

### Agent Development
1. **Query Analysis** - Understand natural language intent
2. **Cypher Template Design** - Create reusable query patterns
3. **State Definition** - Define LangGraph state model
4. **Node Implementation** - Build individual workflow nodes
5. **Integration Testing** - Test end-to-end with sample questions

---

## 7. Data Quality & Validation Rules

### JSON Output Validation
*   Schema compliance: Validate against expected structure
*   Entity uniqueness: Check for duplicate node IDs
*   Relationship integrity: Ensure start/end node IDs exist
*   Property consistency: Validate required properties are present

### Neo4j Data Validation
*   Run count queries after ingestion
*   Verify relationship directions
*   Check for orphaned nodes
*   Validate property types

---

## 8. Testing & Verification

### Pre-Deployment Checks
- [ ] Run all pipeline scripts sequentially without errors
- [ ] Verify Neo4j data with sample Cypher queries
- [ ] Test agent with 3+ diverse natural language questions
- [ ] Check for LLM hallucinations in answers
- [ ] Validate Cypher syntax correctness

### Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| `CypherSyntaxError` | Refine Cypher generation prompt with schema context |
| `KeyError` in prompts | Validate all template variables are populated |
| Invalid JSON from LLM | Add explicit JSON format instructions in prompt |
| Empty query results | Check relationship types and node labels match schema |

---

## 9. VIBE Coding Stability

*   **Incremental Changes:** Small, testable modifications
*   **No Prompt Hardcoding:** Use config or prompt files
*   **Explicit Error Handling:** Never silent failures
*   **Logging:** Log all LLM calls, Cypher queries, and errors
*   **Version Control:** Commit working states before major refactors
*   **No Assumptions:** Always verify Neo4j connection before queries

---

## 10. Docker & Infrastructure

### Required Services (via `docker-compose.yml`)
*   **Neo4j:** Port 7687 (Bolt), 7474 (Browser)
*   **PostgreSQL with pgvector:** Port 5432 (optional, for future vector search)

### Connection Validation
Always verify service health before running scripts:
```bash
# Check Neo4j connection
docker exec db_neo4j cypher-shell -u admin -p admin123 "RETURN 1"
```

---

## 11. Version Tracking

When creating new versions of scripts:
*   Keep previous versions (`_v1`, `_v2`, `_v3`) for reference
*   Document major changes in comments at file top
*   Update README.md usage instructions