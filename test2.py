import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Verify NumPy version
import numpy
print("NumPy version:", numpy.__version__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Marie Curie was born in Poland in 1867.",
    "She won the Nobel Prize in Physics.",
    "She later also won a Nobel Prize in Chemistry.",
    "Pierre Curie was her husband and collaborator.",
    "She became a professor at the University of Paris."
]

# Convert documents to embeddings
embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Query
query = "What awards did Marie Curie win?"
query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# Search top-k
top_k = 3
distances, indices = index.search(query_vec, top_k)

# Retrieve top documents as context
context = "\n".join([documents[i] for i in indices[0]])

# Call DeepSeek
client = openai.OpenAI(
    api_key="", 
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

# Print the LLM's answer
print("\nAnswer:")
print(response.choices[0].message.content)
