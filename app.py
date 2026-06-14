import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Load data
index = faiss.read_index("ncert_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search(query, top_k=5):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector)
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    return [chunks[idx] for idx in indices[0]]

def ask(question):
    relevant_chunks = search(question)
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful NCERT Science Teaching Assistant for Class 9 and 10 students."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("RAG Based AI Teaching Assistant")
st.subheader("NCERT Science — Class 9 & 10")
st.write("Ask any Science question from your NCERT textbook!")

question = st.text_input("Your Question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Finding answer..."):
            answer = ask(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question!")