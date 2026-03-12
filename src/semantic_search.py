import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self, questions):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.questions = questions
        self.embeddings = self.model.encode(questions)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def search(self, query, k=5):
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_emb).astype("float32"), k
        )
        results = []
        for idx in indices[0]:
            results.append(self.questions[idx])
        return results
