import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Config
embedding_model = "all-MiniLM-L6-v2"
index_file = "faiss_fiqa_200.index"
metadata_file = "fiqa_200_docs.pkl"
top_k = 10

# Load model
model = SentenceTransformer(embedding_model)

# Load FAISS index
index = faiss.read_index(index_file)

# Load document metadata
with open(metadata_file, "rb") as f:
    metadata = pickle.load(f)
documents = metadata["texts"]
doc_ids = metadata["doc_ids"]
# Define your query
query = "Most common types of financial scams an individual investor should beware of?"

def retrieve_top_k(query: str, top_k: int = 10):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    return [(doc_ids[i], distances[0][rank]) for rank, i in enumerate(indices[0])]
'''
# Encode query
query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

# Perform similarity search
top_k = 10
distances, indices = index.search(query_vec, top_k)
top_idx = indices[0][0]

# Return the most relevant document and its ID
top_document = documents[top_idx]
top_doc_id = doc_ids[top_idx]

print(f"\nQuery: {query}\n")
print(f"Top {top_k} Relevant Documents:\n")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"[{doc_ids[idx]}#{rank}] similarity={dist:.4f}")'''


# this is used to print the top k documents
'''
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"[{rank}] Doc ID: {doc_ids[idx]}")
    print(f"     Distance: {dist:.4f}")
    print(f"     Text: {documents[idx]}\n")'''