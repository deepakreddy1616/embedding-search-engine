import pickle
import os

class CacheManager:
    """Manages embedding cache using pickle"""
    
    def __init__(self, path="data/cache/embeddings.pkl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.cache = self._load()

    def _load(self):
        """Load cache from disk"""
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        """Save cache to disk"""
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)

    def get(self, doc_id, doc_hash):
        """Get embedding from cache if hash matches"""
        entry = self.cache.get(doc_id)
        if entry and entry["hash"] == doc_hash:
            return entry["embedding"]
        return None

    def set(self, doc_id, doc_hash, embedding):
        """Store embedding in cache"""
        self.cache[doc_id] = {"hash": doc_hash, "embedding": embedding}
        self.save()
