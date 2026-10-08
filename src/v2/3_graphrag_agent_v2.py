"""
GraphRAG Agent v2.0 - Metadata-aware Vector Search (No Text2Cypher)
- Query Rewriting (LLM)
- Vector Search with Metadata Filtering (Neo4j)
- Reranking (LLM)
- Answer Generation
"""

import os
import json
import time
import re
import traceback
import sys
import requests
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

from neo4j import GraphDatabase
import openai
from dotenv import load_dotenv
from tqdm import tqdm

import config

# ============================================================
# Client Initialization
# ============================================================
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.MODEL_API_URL
)

driver = GraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# ============================================================
# Data Models
# ============================================================

class SearchFilters(BaseModel):
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_number: Optional[str] = None # e.g. "S1E01"
    entity_names: List[str] = Field(default_factory=list)
    relationship_types: List[str] = Field(default_factory=list)

class RewrittenQuery(BaseModel):
    original_query: str
    search_queries: List[str] # Queries optimized for vector search
    filters: SearchFilters
    reasoning: str

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str # "node" or "relationship"

# ============================================================
# 1. Query Rewriter (LLM)
# ============================================================

REWRITE_SYSTEM_PROMPT = f"""You are a search query optimizer for a Demon Slayer knowledge graph.
Your goal is to analyze the user's question and convert it into a structured search plan.

## Available Metadata Filters:
- season (int): 1, 2, ...
- episode (int): 1, 2, ...
- episode_number (str): "S1E01", "S1E02", ...
- entity_names (list): {', '.join(config.NODE_LABELS)} (Characters like 'Kamado Tanjiro', 'Nezuko', etc.)
- relationship_types (list): {', '.join(config.RELATIONSHIP_TYPES)}

## Instructions:
1. **Search Queries**: specific keywords for vector search.
    - Example: "Tanjiro fights Rui" -> ["Tanjiro vs Rui battle", "Hinokami Kagura technique"]
2. **Filters**: Extract constraints.
    - "S1E19" -> season: 1, episode: 19, episode_number: "S1E19"
    - "Tanjiro and Nezuko" -> entity_names: ["카마도 탄지로", "카마도 네즈코"] (Use Korean names if possible)
3. **Reasoning**: Briefly explain your strategy.

## Korean Name Mapping (Reference):
- Tanjiro -> 카마도 탄지로
- Nezuko -> 카마도 네즈코
- Zenitsu -> 아가츠마 젠이츠
- Inosuke -> 하시비라 이노스케
- Giyu -> 토미오카 기유
- Muzan -> 키부츠지 무잔
- Rui -> 루이
- Rengoku -> 렌고쿠 쿄쥬로

Output must be JSON.
"""

def rewrite_query(user_query: str) -> RewrittenQuery:
    """Rewrite user query into vector search queries and filters"""
    try:
        completion = client.beta.chat.completions.parse(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            response_format=RewrittenQuery,
            temperature=0
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"❌ Query Rewrite Failed: {e}")
        # Fail fast if connection error
        if "Connection error" in str(e) or "Connection refused" in str(e):
            raise e
        # Fallback
        return RewrittenQuery(
            original_query=user_query,
            search_queries=[user_query],
            filters=SearchFilters(),
            reasoning="Fallback due to error"
        )

# ============================================================
# 2. Vector Retriever (Neo4j)
# ============================================================

def get_embedding(text: str) -> List[float]:
    start = time.time()
    try:
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[text]
        )
        # print(f"  ⏱️ Embedding ({len(text)} chars): {time.time()-start:.2f}s")
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        # Fail fast if connection error
        if "Connection error" in str(e) or "Connection refused" in str(e):
            raise e
        return [0.0] * config.EMBEDDING_DIMENSION

def build_filter_clause(filters: SearchFilters, variable: str = "node") -> str:
    """Build Cypher WHERE clause from filters"""
    conditions = []
    
    # Season/Episode filter
    if filters.season is not None:
        conditions.append(f"{variable}.season = {filters.season}")
    if filters.episode is not None:
        conditions.append(f"{variable}.episode = {filters.episode}")
    if filters.episode_number is not None:
        conditions.append(f"{variable}.episode_number = '{filters.episode_number}'")
    
    # Entity Name Filter (for relationships, check start/end node names?)
    # For now, simplistic approach: if we are querying nodes, filter by name
    # If querying relationships, maybe filter by related node names if possible, but vector search handles context.
    
    return " AND ".join(conditions) if conditions else "1=1" # 1=1 is always true

def vector_search(
    rewritten: RewrittenQuery, 
    top_k_per_query: int = 20
) -> List[SearchResult]:
    
    results = []
    seen_ids = set()
    
    with driver.session(database=config.NEO4J_DATABASE) as session:
        for query_text in rewritten.search_queries:
            embedding = get_embedding(query_text)
            
            # A. Entity Search
            for label in config.NODE_LABELS:
                index_name = f"{config.VECTOR_INDEX_NODE}_{label}"
                cypher = f"""
                CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                YIELD node, score
                RETURN node, score
                """
                # Note: Node indices might not have season/episode props, so skipping filters for nodes usually
                # Unless nodes represent 'Events'. Here nodes are Characters.
                
                try:
                    res = session.run(cypher, index_name=index_name, k=5, embedding=embedding) # Low k for entities
                    for record in res:
                        node = record["node"]
                        score = record["score"]
                        
                        # Filter by entity name if specified
                        if rewritten.filters.entity_names:
                            if node.get("name") not in rewritten.filters.entity_names:
                                continue

                        content = f"Character: {node.get('name')} ({node.get('label')})"
                        res_obj = SearchResult(
                            content=content, 
                            score=score, 
                            metadata=dict(node),
                            source="node"
                        )
                        if node.element_id not in seen_ids:
                            results.append(res_obj)
                            seen_ids.add(node.element_id)
                except Exception as e:
                    # print(f"⚠️ Entity search error ({index_name}): {e}")
                    pass

            # B. Relationship Search (The Core)
            # Filter clauses
            filter_clause = build_filter_clause(rewritten.filters, variable="r")
            
            # If relationship_types specified in filter, only query those indices
            target_rels = rewritten.filters.relationship_types if rewritten.filters.relationship_types else config.RELATIONSHIP_TYPES
            
            # Limit target rels to avoid too many queries? 
            # Strategy: Query ALL valid relationship indices using UNION logic or loop
            # given the number of types (20+), looping all might be slow.
            # Optimization: 2_ingest_data creates separate index per type.
            
            # Let's enforce a limit or priority. 
            # Or use a consolidated index approach? No, we stuck to separate indices.
            # We will query ALL relevant indices.
            
            for rel_type in target_rels:
                index_name = f"{config.VECTOR_INDEX_RELATIONSHIP_PREFIX}_{rel_type}"
                
                cypher = f"""
                CALL db.index.vector.queryRelationships($index_name, $k, $embedding)
                YIELD relationship as r, score
                WHERE {filter_clause}
                MATCH (start)-[r]->(end)
                RETURN r, start.name as start_name, end.name as end_name, type(r) as type, score
                """
                
                try:
                    # print(f"Querying index: {index_name}")
                    res = session.run(cypher, index_name=index_name, k=top_k_per_query, embedding=embedding)
                    
                    for record in res:
                        r = record["r"]
                        score = record["score"]
                        
                        # Construct rich content
                        props = dict(r)
                        desc = props.get("description") or props.get("context") or ""
                        content = f"[{record['type']}] {record['start_name']} -> {record['end_name']}: {desc} (Ep: {props.get('episode_number')})"
                        
                        res_obj = SearchResult(
                            content=content,
                            score=score,
                            metadata=props,
                            source="relationship"
                        )
                        
                        # Dedup by element_id
                        if r.element_id not in seen_ids:
                            results.append(res_obj)
                            seen_ids.add(r.element_id)
                            
                except Exception as e:
                    # print(f"⚠️ Rel search error ({index_name}): {e}")
                    pass
    
    # Sort all results by score
    results.sort(key=lambda x: x.score, reverse=True)
    return results

# ============================================================
# 3. Reranker (LLM)
# ============================================================

class RerankedResult(BaseModel):
    index: int
    relevance_score: float # 0.0 to 1.0
    reasoning: str

class RerankResponse(BaseModel):
    ranked_indices: List[RerankedResult]

def rerank_results(query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
    """Rerank results using LLM"""
    if not results:
        return []
        
    candidates = results[:config.VECTOR_TOP_K * 2] # Rerank top 50-100 inputs
    
    # Prepare items for LLM
    items_text = ""
    for i, res in enumerate(candidates):
        items_text += f"[{i}] {res.content}\n"

    system_prompt = """You are a relevance ranking system.
1. Analyze the user query and the list of retrieved items.
2. Assign a relevance score (0.0 - 1.0) to each item based on how well it answers the query.
3. Return the indices of the top most relevant items (max 10-15).
"""
    
    try:
        completion = client.beta.chat.completions.parse(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nItems:\n{items_text}"}
            ],
            response_format=RerankResponse,
            temperature=0
        )
        
        ranked_indices = completion.choices[0].message.parsed.ranked_indices
        
        # Sort candidates based on LLM scores
        # Map back to objects
        reranked = []
        for r_item in ranked_indices:
            if 0 <= r_item.index < len(candidates):
                obj = candidates[r_item.index]
                # Update score with LLM relevance score
                obj.score = r_item.relevance_score
                reranked.append(obj)
        
        # Sort descending
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
        
    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        return candidates[:top_k]

# ============================================================
# 4. Answer Generator
# ============================================================

ANSWER_PROMPT = """Based on the context provided below, answer the user's question.

## User Question:
{question}

## Context:
{context}

## Guidelines:
1. Use ONLY the provided context. If unsure, say "I don't have enough information".
2. Organize the answer logically (e.g., by Episode or Event).
3. Be concise but comprehensive.
4. Answer in Korean.

## Answer:
"""

def generate_answer(query: str, context_items: List[SearchResult]) -> str:
    context_str = "\n\n".join([item.content for item in context_items])
    
    prompt = ANSWER_PROMPT.format(question=query, context=context_str)
    
    start = time.time()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    # print(f"  ⏱️ Generation: {time.time()-start:.2f}s")
    return resp.choices[0].message.content

# ============================================================
# Main Pipeline
# ============================================================

def check_services():
    """Check if Neo4j and Ollama are reachable"""
    print("🔍 Checking services...")
    
    # 1. Neo4j
    try:
        driver.verify_connectivity()
        print("  ✅ Neo4j is reachable")
    except Exception as e:
        print(f"  ❌ Neo4j is NOT reachable: {e}")
        sys.exit(1)

    # 2. Ollama
    try:
        # Extract base URL (remove /v1)
        base_url = config.MODEL_API_URL.replace("/v1", "")
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            print("  ✅ Ollama is reachable")
            # Optional: Check if required models are pulled
            models = [m['name'] for m in resp.json()['models']]
            required_models = [config.LLM_MODEL.split(':')[0], config.EMBEDDING_MODEL.split(':')[0]]
            # diverse checking logic... let's just warn for now
        else:
            print(f"  ⚠️ Ollama returned status {resp.status_code}")
    except Exception as e:
        print(f"  ❌ Ollama is NOT reachable at {base_url}: {e}")
        print("  💡 Hint: Ensure 'docker-compose up -d ollama' is running.")
        sys.exit(1)

def graphrag_pipeline(user_query: str):
    check_services()

    print("\n" + "="*80)
    print(f"🚀 Processing: {user_query}")
    print("="*80)
    
    # 1. Rewrite
    print(f"\n🤔 Rewriting Query...")
    rewritten = rewrite_query(user_query)
    print(f"  ✅ Queries: {rewritten.search_queries}")
    print(f"  ✅ Filters: {rewritten.filters}")
    
    # 2. Retrieval
    print(f"\n🔍 Vector Search (Top-K per query: 20)...")
    raw_results = vector_search(rewritten)
    print(f"  ✅ Retrieved {len(raw_results)} candidates")
    
    # 3. Reranking
    print(f"\n⚖️  Reranking (Target Top-K: {config.VECTOR_TOP_K})...")
    final_results = rerank_results(user_query, raw_results, top_k=config.VECTOR_TOP_K)
    print(f"  ✅ Selected {len(final_results)} items")
    
    for i, res in enumerate(final_results):
        print(f"    [{i+1}] {res.score:.2f} | {res.content[:80]}...")
        
    if not final_results:
        return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
    # 4. Generate
    print(f"\n✍️  Generating Answer...")
    answer = generate_answer(user_query, final_results)
    
    return answer

if __name__ == "__main__":
    test_queries = [
        "카마도 탄지로와 카마도 네즈코 사이에 어떤 사건들이 있었어?",
        "S1E19에서 탄지로는 누구와 싸웠어?",
        "토미오카 기유의 역할은?"
    ]
    
    for q in test_queries:
        result = graphrag_pipeline(q)
        print("\n" + "-"*80)
        print(f"📝 Final Answer:\n{result}")
        print("-"*80 + "\n")
