import json
import logging
from typing import List, Dict, Tuple
import openai
from sentence_transformers import SentenceTransformer
from kg_forest_new import KnowledgeGraphBuilder, DeepSeekExtractor, Entity, Relation, ChunkExtraction
import numpy as np
import sys
import pandas as pd
from datasets import load_from_disk

# Global variables
DATASET_PATH = "datasets/MultiHopRAG/train.json"
API_KEY = ""
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = "https://api.deepseek.com/v1"
NUM_QUERIES_TO_PROCESS = 3  # Number of queries to process from the dataset

# use this when dynamic creating kg
#kg_builder = KnowledgeGraphBuilder(api_key=API_KEY, model_name=MODEL_NAME)

# for saved
kg_builder = KnowledgeGraphBuilder.load_graph(api_key=API_KEY, path_prefix="saved/graph")

rel_extractor = kg_builder.extractor
rel_extractor.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def paraphrase_query(query: str) -> List[str]:
    #(f"Paraphrasing query: {query}")
    try:
        resp = rel_extractor.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "You are a paraphrasing assistant.\n"
                    "Given a query, generate exactly 3 clean, fluent paraphrases.\n"
                    "Rules:\n"
                    "- Only return the 3 paraphrased sentences, nothing else.\n"
                    "- No introductions, no commentary, no alternative notes.\n"
                    f"Query: \"{query}\""
                )}
            ],
            temperature=0.0
        )
        paraphrases = resp.choices[0].message.content.strip().split("\n")
        #print(f"Paraphrased variants: {paraphrases}")
        return [p.strip() for p in paraphrases if p.strip()]
    except Exception as e:
        logging.error(f"Error paraphrasing query: {str(e)}")
        return []
    
def extract_entities_and_relations(query: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    paraphrases = paraphrase_query(query)
    all_entities: List[Dict[str, str]] = []
    all_relations: List[Dict[str, str]] = []

    for variant in paraphrases:
        result = rel_extractor.extract_chunk(variant)
        #print(json.dumps({
            #"entities": [vars(e) for e in result.entities],
           # "relations": [vars(r) for r in result.relations]
        #}, indent=2, ensure_ascii=False))


        entities = [vars(e) for e in result.entities]
        relations = [vars(r) for r in result.relations]

        all_entities.extend(entities)
        all_relations.extend(relations)

    #print(f"\nCollected Entities: {json.dumps(all_entities, indent=2, ensure_ascii=False)}")
    #print(f"\nCollected Relations: {json.dumps(all_relations, indent=2, ensure_ascii=False)}")

    return all_entities, all_relations

from typing import Optional, List, Dict, Tuple

def search_entities_with_optional_relations(
    kg_builder: KnowledgeGraphBuilder,
    entities: List[Dict[str, str]],
    relations: Optional[List[Dict[str, str]]] = None,
    top_k: int = 10,
    max_hops: int = 3,                # 新增参数：最多跳数，默认2跳
) -> List[Tuple[str, List[str]]]:
    """
    Search for relevant chunks using entities and optionally relations,
    with up to max_hops hops.
    """
    entity_embedder = kg_builder.entity_embedder
    relation_embedder = kg_builder.relation_embedder
    index = kg_builder.index

    all_results = []

    for ent in entities:
        start_entity = ent["entity_name"]
        combined_chunk_ids = set()
        # frontier 存放当前 hop 要搜索的实体名
        frontier = [start_entity]
        visited = {start_entity}

        for hop in range(max_hops):
            next_frontier = []
            for ent_name in frontier:
                # embedding
                ent_vec = entity_embedder.encode(
                    ent_name,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype(np.float32)

                # 第一跳时，同时做 relation 搜索
                if hop == 0 and relations:
                    rel = next((r for r in relations if r["target_entity"] == ent_name), None)
                    if rel:
                        rel_vec = relation_embedder.encode(
                            rel["relation"],
                            normalize_embeddings=True,
                            convert_to_numpy=True
                        ).astype(np.float32)
                        for _, chunk_list in index.search_by_name(ent_vec, rel_vec, top_k=top_k):
                            combined_chunk_ids.update(chunk_list)

                # 实体-only 搜索
                for _, chunk_list in index.search(ent_vec, top_k=top_k):
                    combined_chunk_ids.update(chunk_list)

                # 为下一跳收集实体（基于所有提取到的 relations）
                if relations:
                    for r in relations:
                        if r["source_entity"] == ent_name and r["target_entity"] not in visited:
                            visited.add(r["target_entity"])
                            next_frontier.append(r["target_entity"])

            # 下一跳
            frontier = next_frontier
            if not frontier:
                break

        all_results.append((start_entity, list(combined_chunk_ids)))

    return all_results

def rank_chunks_by_similarity(
    kg_builder,
    query_text: str,
    chunk_ids: List[str],
    top_k: int = 10
) -> List[Dict]:
    """
    Given a query and chunk IDs, return the top-k most similar chunks based on cosine similarity.

    Returns:
        List of dicts with keys: chunk_id, text, embedding, similarity
    """
    # Step 1: Embed the query
    query_vec = kg_builder.entity_embedder.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    results = []

    for cid in chunk_ids:
        chunk_vec = kg_builder.embeddings.get(cid)
        if chunk_vec is None:
            continue

        similarity = np.dot(query_vec, chunk_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
        )

        results.append({
            "chunk_id": cid,
            "embedding": chunk_vec,
            "similarity": similarity
        })

    # Step 2: Sort by similarity and return top-k
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]

def display_chunk_content(kg_builder, chunk_id: str) -> None:
    """
    Display the content of a specific chunk.
    
    Args:
        kg_builder: KnowledgeGraphBuilder instance
        chunk_id: The ID of the chunk to display
    """
    if chunk_id in kg_builder.extractions:
        # Get the chunk metadata
        meta = kg_builder.chunk_meta.get(chunk_id, {})
        logging.info(f"Chunk metadata for {chunk_id}: {meta}")
        print(f"\nChunk ID: {chunk_id}")
        print(f"Document ID: {meta.get('doc', 'Unknown')}")
        print(f"Title: {meta.get('title', 'Unknown')}")
        print(f"Position: {meta.get('start', 0)}-{meta.get('end', 0)}")
        
        # Get the entities and relations
        # extraction = kg_builder.extractions[chunk_id]
        # print("\nEntities:")
        # for entity in extraction.entities:
        #     print(f"- {entity.entity_name} ({entity.entity_type}): {entity.entity_description}")
        
        # print("\nRelations:")
        # for relation in extraction.relations:
        #     print(f"- {relation.source_entity} --[{relation.relation}]--> {relation.target_entity}")
        #     print(f"  Description: {relation.relation_description}")
    else:
        print(f"\nChunk {chunk_id} not found in knowledge graph")

def main():
    """Main function: Demonstrates how to use the knowledge graph query system"""
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S',
                       level=logging.INFO)
    
    # Load saved knowledge graph
    kg_builder = KnowledgeGraphBuilder.load_graph(api_key=API_KEY, path_prefix="saved/graph")
    
    # Load first three queries from MultiHopRAG dataset
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process queries
    for i in range(min(NUM_QUERIES_TO_PROCESS, len(data))):
        query_data = data[i]
        query = query_data['query']
        expected_answer = query_data['answer']
        
        print(f"\n{'='*80}")
        print(f"Query {i+1}:")
        print(f"Question: {query}")
        print(f"Expected Answer: {expected_answer}")
        print(f"{'='*80}")
        
        # Extract entities and relations from query
        entities, relations = extract_entities_and_relations(query)
        
        # Execute search with entities and relations
        results = search_entities_with_optional_relations(kg_builder, entities, relations)
        
        # Print results
        print("\nSearch Results:")
        all_chunks = set()  # Initialize set to collect all chunks
        for entity, chunks in results:
            print(f"{entity}: {chunks}")
            all_chunks.update(chunks)  # Add chunks to the set
        
        # Rank and print chunks by similarity
        if all_chunks:
            print("\nRanked Results by Similarity:")
            ranked_chunks = rank_chunks_by_similarity(kg_builder, query, list(all_chunks))
            for chunk in ranked_chunks:
                meta = kg_builder.chunk_meta.get(chunk['chunk_id'], {})
                print(f"[{chunk['chunk_id']}] similarity={chunk['similarity']:.3f} Title: {meta.get('title', 'Unknown')}")
            
            # Display content of each chunk
            # print("\nDetailed Chunk Contents:")
            # for chunk in ranked_chunks:
            #     display_chunk_content(kg_builder, chunk['chunk_id'])
        
        print("\n")

if __name__ == "__main__":
    main()