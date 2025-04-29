import os
import json
import re
import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
import openai
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path
import torch
import shutil
from sentence_transformers import SentenceTransformer


# === Assume you import your models ===
from kg_forest import KGProcessor
from kg_forest import RelationExtractor
STORAGE_PATH = "data"
API_KEY = "sk-2c14fbdaec1645189872267405e3d6a5"
BASE_URL = "https://api.deepseek.com/v1"

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
processor = KGProcessor(STORAGE_PATH, API_KEY, BASE_URL)
rel_extractor = processor.relation_extractor
rel_extractor.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL) 
kg = processor.kg


# given query, paragrphrase
def paraphrase_query(query: str) -> List[str]:
    """Paraphrase a query into 3 variants."""
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

# given pharaphrased queries, find entities and attributes 
def extract_entities_and_relations(query: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Search for the top-k most similar chunks to the query.
    """
    # 1. Paraphrase the query
    paraphrases = paraphrase_query(query)
    #paraphrases = ['Which organizations were associated with the group of breast cancer patients?', 'What groups were connected to the breast cancer patient cohort?', 'To which organizations was the breast cancer patient cohort tied?']
    all_entities: List[Dict[str, str]] = []
    all_relations: List[Dict[str, str]] = []

    for var in paraphrases:
        extraction = rel_extractor.extract_from_chunk(var)
        print(json.dumps(extraction, indent=2, ensure_ascii=False))
        '''{
            "entities": [
                {
                "entity_name": "breast cancer cohort",
                "entity_type": "Medical/Research Group",
                "entity_description": "A group or study focused on breast cancer"
                }
            ],
            "relations": [
                {
                "source_entity": "Unknown organizations",
                "relation": "participated_in",
                "target_entity": "breast cancer cohort",
                "relation_description": "Organizations that took part in the breast cancer cohort"
                }
            ]
            }'''
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        all_entities.extend(entities)
        all_relations.extend(relations)
    # print entity and relation lists
    print(f"\nCollected Entities: {json.dumps(all_entities, indent=2, ensure_ascii=False)}")
    print(f"\nCollected Relations: {json.dumps(all_relations, indent=2, ensure_ascii=False)}")

    return all_entities, all_relations

if __name__ == "__main__":
    # for actual use
    #query = "What organizations were involved in the breast cancer patient cohort?"
    #entities, relations = extract_entities_and_relations(query)

    # for testing
    entities = [
        {
            "entity_name": "study cohort",
            "entity_type": "Event",
            "entity_description": "A cohort study involving patients."
        },
        {
            "entity_name": "breast cancer patients",
            "entity_type": "Group",
            "entity_description": "Patients diagnosed with breast cancer."
        }
    ]

    relations = [
        {
            "source_entity": "study cohort",
            "relation": "includes",
            "target_entity": "breast cancer patients",
            "relation_description": "The study cohort includes breast cancer patients."
        }
    ]


    all_chunks = []  # ← to store all retrieved chunks

    for entity_info in entities:
        entity_name = entity_info["entity_name"]

        # Find all relations that have this entity as a target
        matching_relations = [r for r in relations if r["target_entity"] == entity_name]

        if matching_relations:
            for rel in matching_relations:
                relation_name = rel["relation"]
                # Lookup specific entity + relation
                #chunks = kg.lookup_entity_relation(entity_name, relation_name=relation_name)
                chunks = processor.lookup_relation_chunk_embeddings(entity_name, relation_name=relation_name)

                all_chunks.extend(chunks)
        else:
            # If no specific relation found, lookup all relations under entity
            chunks = kg.lookup_entity_relation(entity_name)
            all_chunks.extend(chunks)

    # Optional: remove duplicates
    all_chunks = list(set(all_chunks))

    # Step 4: Now you have all chunks!
    print(f"Collected {len(all_chunks)} unique chunks:")
    print(all_chunks)