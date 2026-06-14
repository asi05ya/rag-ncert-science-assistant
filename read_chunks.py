import requests

r = requests.post("http://localhost:11434/api/embeddings", json={
              "model": "bge-m3",
              "prompt": "asiya is good  girl"
})

print(r.json())