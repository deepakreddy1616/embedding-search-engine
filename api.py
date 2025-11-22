from fastapi import FastAPI
from pydantic import BaseModel
from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import SearchEngine
import numpy as np

app = FastAPI(title="Embedding Search Engine API")

# Initialize components
print("Initializing search engine...")
processor = DocumentPreprocessor()
documents = processor.process_dir("data/docs")
embedder = EmbeddingGenerator()

# Generate embeddings for all documents
print("Generating embeddings...")
doc_ids = []
embeddings_list = []

for doc in documents:
    doc_id = doc["metadata"]["filename"]
    doc_hash = doc["metadata"]["hash"]
    text = doc["text"]
    
    emb = embedder.embed(doc_id, doc_hash, text)
    doc_ids.append(doc_id)
    embeddings_list.append(emb)

embeddings = np.array(embeddings_list)
search_engine = SearchEngine(doc_ids, embeddings, documents)

print(f"✅ Search engine ready with {len(documents)} documents")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "Embedding Search Engine API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "total_documents": len(doc_ids),
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }

@app.post("/search")
def search(req: QueryRequest):
    """Search for similar documents"""
    # Generate query embedding
    q_emb = embedder.model.encode(req.query, convert_to_numpy=True)
    
    # Search
    results = search_engine.search(q_emb, req.top_k)
    
    return {"results": results, "query": req.query}
