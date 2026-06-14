from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Read the text file
with open("output.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = split_text(text)
print(f"Total chunks: {len(chunks)}")

# Load the embedding model
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
print("Creating embeddings... Please wait ⏳")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings)

# Normalize for Cosine Similarity
faiss.normalize_L2(embeddings)

# Create FAISS index with Inner Product (Cosine Similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and chunks
faiss.write_index(index, "ncert_index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Done! Cosine Similarity vectors saved successfully!")