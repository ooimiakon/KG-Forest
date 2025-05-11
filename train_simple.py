import os
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Config
corpus_file = "./fiqa/corpus.jsonl"
max_docs = 200
embedding_model = "all-MiniLM-L6-v2"
index_file = "faiss_fiqa_200.index"
metadata_file = "fiqa_200_docs.pkl"

# Load documents
documents = []
doc_ids = []

with open(corpus_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i > max_docs:
            break
        record = json.loads(line)
        doc_ids.append(record["_id"])
        documents.append(record["text"])

print(f"Loaded {len(documents)} documents.")

# Load embedding model
model = SentenceTransformer(embedding_model)

# Encode documents
embeddings = model.encode(
    documents, convert_to_numpy=True, normalize_embeddings=True
).astype(np.float32)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, index_file)

# Save associated document metadata
with open(metadata_file, "wb") as f:
    pickle.dump({"doc_ids": doc_ids, "texts": documents}, f)

print(f"Saved FAISS index with {index.ntotal} documents to '{index_file}'.")
print(f"Saved document metadata to '{metadata_file}'.")
