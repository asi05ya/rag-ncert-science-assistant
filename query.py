from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load saved data
index = faiss.read_index("ncert_index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=5):
    # Convert query to vector
    query_vector = model.encode([query])
    query_vector = np.array(query_vector)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_vector)
    
    # Search in FAISS
    scores, indices = index.search(query_vector, top_k)
    
    # Return results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "score": scores[0][i],
            "text": chunks[idx]
        })
    return results

# Test
query = "What is photosynthesis?"
results = search(query)

print(f"Query: {query}\n")
for i, result in enumerate(results):
    print(f"Result {i+1} (Score: {result['score']:.4f}):")
    print(result['text'])
    print("---")