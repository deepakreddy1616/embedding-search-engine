from sentence_transformers import SentenceTransformer
from .cache_manager import CacheManager

class EmbeddingGenerator:
    """Generates embeddings with intelligent caching"""
    
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.cache = CacheManager()
        print("Model loaded successfully")

    def embed(self, doc_id, doc_hash, text):
        """Generate or retrieve cached embedding"""
        # Check cache first
        cached = self.cache.get(doc_id, doc_hash)
        if cached is not None:
            return cached
        
        # Generate new embedding
        emb = self.model.encode(text, convert_to_numpy=True)
        
        # Store in cache
        self.cache.set(doc_id, doc_hash, emb)
        return emb
