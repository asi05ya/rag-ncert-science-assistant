from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load saved data
index = faiss.read_index("ncert_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
client = Groq(api_key=api_key)

def search(query, top_k=5):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector)
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append(chunks[idx])
    return results

def ask(question):
    relevant_chunks = search(question)
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful NCERT Science Teaching Assistant
                for Class 9 and 10 students. Answer questions based on the
                provided context from NCERT textbooks only."""
            },
            {
                "role": "user",
                "content": f"""Context from NCERT textbook:
{context}

Question: {question}

Please provide a clear and simple answer based on the context above."""
            }
        ]
    )
    return response.choices[0].message.content

# Main
print("RAG Based AI Teaching Assistant - NCERT Science")
print("=" * 50)
question = input("Ask your question: ")
print(f"\nQuestion: {question}\n")
answer = ask(question)
print(f"Answer:\n{answer}")