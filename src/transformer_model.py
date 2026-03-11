from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import n um py as np

class  TransformerSimilarity:
    def __init__(self,thresold=0.7):
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
        self.thresold=thresold
    

    def predict(self,q1_list,q2_list):
        emb1=self.model.encode(q1_list,show_progress_bar=True)
        emb2=self.model.encode(q2_list,show_progress_bar=True)
        sims=cosine_similarity(emb1,emb2).diagonal()
        return (sims>self.thresold).astype(int)

    def similarity(self,q1,q2):
        emb1=self.model.encode([q1])
        emb2=self.model.encode([q2])
        sim=cosine_similarity(emb1,emb2)[0][0]
        return sim