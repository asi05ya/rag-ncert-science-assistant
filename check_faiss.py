import faiss
import pickle
import numpy as np

# FAISS index load karo
index = faiss.read_index("ncert_index.faiss")
print(f"✅ FAISS Index loaded!")
print(f"Total vectors: {index.ntotal}")

# Chunks load karo
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"Total chunks: {len(chunks)}")

# Example chunk dekho
print("\n--- Example Chunk ---")
print(chunks[10])