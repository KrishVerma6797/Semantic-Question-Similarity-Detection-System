import os
import numpy as np
from sentence_transformers import SentenceTransformer

class TransformerSimilarity:

    def __init__(self, threshold=0.7):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    # Used for dataset evaluation
    def predict(self, q1_list, q2_list):

        if os.path.exists("q1_emb.npy") and os.path.exists("q2_emb.npy"):
            emb1 = np.load("q1_emb.npy")
            emb2 = np.load("q2_emb.npy")
        else:
            emb1 = self.model.encode(q1_list, show_progress_bar=True)
            emb2 = self.model.encode(q2_list, show_progress_bar=True)

            np.save("q1_emb.npy", emb1)
            np.save("q2_emb.npy", emb2)

        sims = np.sum(emb1 * emb2, axis=1)

        return (sims > self.threshold).astype(int)

    # Used for interactive testing (two questions)
    def similarity(self, q1, q2):
        emb1 = self.model.encode([q1])
        emb2 = self.model.encode([q2])

        sim = np.sum(emb1 * emb2, axis=1)[0]
        return sim
