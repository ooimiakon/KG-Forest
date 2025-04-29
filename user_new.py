import json
import logging
from typing import List, Dict, Tuple
import openai
from sentence_transformers import SentenceTransformer
from kg_forest_new import KnowledgeGraphBuilder, DeepSeekExtractor, Entity, Relation, ChunkExtraction
import numpy as np


API_KEY = ""
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = "https://api.deepseek.com/v1"
# use this when dynamic creating kg
#kg_builder = KnowledgeGraphBuilder(api_key=API_KEY, model_name=MODEL_NAME)

# for saved
kg_builder = KnowledgeGraphBuilder.load_graph(api_key=API_KEY, path_prefix="saved/graph")

rel_extractor = kg_builder.extractor
rel_extractor.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def paraphrase_query(query: str) -> List[str]:
    print(f"Paraphrasing query: {query}")
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
        print(f"Paraphrased variants: {paraphrases}")
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
        print(json.dumps({
            "entities": [vars(e) for e in result.entities],
            "relations": [vars(r) for r in result.relations]
        }, indent=2, ensure_ascii=False))


        entities = [vars(e) for e in result.entities]
        relations = [vars(r) for r in result.relations]

        all_entities.extend(entities)
        all_relations.extend(relations)

    print(f"\nCollected Entities: {json.dumps(all_entities, indent=2, ensure_ascii=False)}")
    print(f"\nCollected Relations: {json.dumps(all_relations, indent=2, ensure_ascii=False)}")

    return all_entities, all_relations

from typing import Optional, List, Dict, Tuple

def search_entities_with_optional_relations(
    kg_builder: KnowledgeGraphBuilder,
    entities: List[Dict[str, str]],
    relations: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5
) -> List[Tuple[str, List[str]]]:
    """
    Search for relevant chunks using entities and optionally relations.

    Args:
        kg_builder: The KnowledgeGraphBuilder instance with built index.
        entities: List of entity dictionaries from extraction.
        relations: Optional list of relation dictionaries from extraction.
        top_k: Top-k similar entities to return.

    Returns:
        A list of (entity_name, [matching chunk_ids])
    """
    entity_embedder = kg_builder.entity_embedder
    relation_embedder = kg_builder.relation_embedder
    if kg_builder.index.entity_index is None:
        raise RuntimeError("FAISS index was not loaded. Make sure `saved/graph.faiss` exists and was saved correctly.")

    index = kg_builder.index

    all_results = []

    for ent in entities:
        ent_name = ent["entity_name"]
        ent_vec = entity_embedder.encode(
            ent_name, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # Find the most relevant relation for this entity (if any)
        rel_vec = None
        if relations:
            # Pick the relation that targets this entity
            rel = next(
                (r for r in relations if r["target_entity"] == ent_name),
                None
            )
            if rel:
                rel_vec = relation_embedder.encode(
                    rel["relation"], normalize_embeddings=True, convert_to_numpy=True
                ).astype(np.float32)

        # Search via relation-aware or entity-only
        if rel_vec is not None:
            results = index.search_by_name(ent_vec, rel_vec, top_k=top_k)
        else:
            results = index.search(ent_vec, top_k=top_k)

        all_results.extend(results)

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


if __name__ == "__main__":
    # chane this for 
    query = "Which organizations were associated with the group of breast cancer patients?"
    #entities, relations = extract_entities_and_relations(query)

    #hard coded for testing
    entities = [
        {'entity_name': 'breast cancer patient group', 'entity_type': 'Organization', 'entity_description': 'Patient group mentioned in the query'},
        {'entity_name': "breast cancer patients' group", 'entity_type': 'Organization', 'entity_description': 'Group mentioned in the query'},
        {'entity_name': 'breast cancer patient collective', 'entity_type': 'Organization', 'entity_description': 'A collective of patients affected by breast cancer'}
    ]

    relations = [
        {'source_entity': 'Unknown organization', 'relation': 'linked_to', 'target_entity': 'breast cancer patient group', 'relation_description': 'Organizations associated with the breast cancer patient group'},
        {'source_entity': 'Unknown organization', 'relation': 'connected_to', 'target_entity': "breast cancer patients' group", 'relation_description': "Organizations linked to the breast cancer patients' group"},
        {'source_entity': 'Unknown groups or associations', 'relation': 'tied_to', 'target_entity': 'breast cancer patient collective', 'relation_description': 'Groups or associations linked to the breast cancer patient collective'}
    ]
    results = search_entities_with_optional_relations(kg_builder, entities, relations)
    print("\nSearch Results:")
    for ent_name, chunk_ids in results:
        print(f"{ent_name}: {chunk_ids}")
    # Step 2: flatten and dedupe chunk IDs
    all_chunk_ids = list({cid for _, chunk_list in results for cid in chunk_list})
    top_chunks = rank_chunks_by_similarity(
    kg_builder,
    query_text= query,
    chunk_ids=all_chunk_ids,
    top_k=10
    )

    # 4. Print
    for chunk in top_chunks:
        print(f"[{chunk['chunk_id']}] similarity={chunk['similarity']:.3f}")

    '''
Search Results:
breast cancer patients: ['MED-10#2']
breast cancer: ['MED-10#1']
breast cancer diagnosis: ['MED-14#2']
cohort of 17,880 breast cancer patients: ['MED-14#0']
breast cancer patients: ['MED-10#0']
cohort of 17,880 breast cancer patients: ['MED-14#0']
breast cancer: ['MED-14#1']
breast cancer diagnosis: ['MED-14#2']
breast cancer patients: ['MED-10#0']
breast cancer: ['MED-10#1']
breast cancer diagnosis: ['MED-14#2']
cohort of 17,880 breast cancer patients: ['MED-14#0']
[MED-14#0] similarity=0.412
[MED-10#0] similarity=0.411
[MED-14#1] similarity=0.385
[MED-14#2] similarity=0.377
[MED-10#1] similarity=0.333
[MED-10#2] similarity=0.323'''
