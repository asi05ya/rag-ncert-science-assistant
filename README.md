# RAG Based AI Teaching Assistant — NCERT Science

An AI-powered teaching assistant that answers Class 9 and 10 
Science questions using Retrieval-Augmented Generation (RAG).

## How It Works
1. NCERT Science PDFs are extracted and converted to text
2. Text is split into chunks for efficient retrieval
3. Chunks are converted to vectors using Sentence Transformers
4. User query is matched with relevant chunks using FAISS
5. Relevant context is sent to Groq LLM for generating answers

## Tech Stack
- Python
- Sentence Transformers
- FAISS (Vector Database)
- Groq API (LLaMA 3.3 70B)
- pdfplumber

## Data Source
NCERT Class 9 and 10 Science Textbooks
Source: ncert.nic.in (Official — Free & Legal)

## Features
- Accurate answers from NCERT textbooks only
- Fast retrieval using Cosine Similarity
- Completely free with Groq API

## Setup Instructions
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Create .env file and add your Groq API key:
   GROQ_API_KEY=your_key_here
4. Run:
   python rag_app.py

## Future Scope
- Diagram extraction from NCERT PDFs
- Hindi language support
- Web interface for students
- Support for more subjects and classes

## Author
Asiya Farzan
B.Com (Computer Applications) — University of Madras
