import pickle

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

with open("output.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = split_text(text)
print(f"Total chunks: {len(chunks)}")

def is_valid_chunk(chunk):
    words = chunk.split()
    if len(words) < 20:
        return False
    single_chars = sum(1 for w in words if len(w) == 1)
    if single_chars / len(words) > 0.3:
        return False
    return True

chunks = [c for c in chunks if is_valid_chunk(c)]
print(f"Clean chunks after filter: {len(chunks)}")

# Save clean chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved successfully!")