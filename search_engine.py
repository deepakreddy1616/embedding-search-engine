from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SearchEngine:
    """Vector search engine using cosine similarity"""
    
    def __init__(self, doc_ids, embeddings, docs):
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.docs = docs
        print(f"Search engine initialized with {len(doc_ids)} documents")

    def search(self, query_emb, top_k=5):
        """Search for most similar documents"""
        # Calculate cosine similarities
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        
        # Get top k indices
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        # Build results
        results = []
        for i in top_idx:
            doc = self.docs[i]
            score = float(sims[i])
            preview = doc['text'][:200] + "..."
            
            # Calculate keyword overlap for explanation
            query_words = set(query_emb)  # placeholder - will improve
            
            results.append({
                "doc_id": self.doc_ids[i],
                "score": round(score, 4),
                "preview": preview,
                "length": doc['metadata']['length']
            })
        return results
