import torch
import torch.nn as nn
import numpy as np


class SimilarityMetrics:
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric
    
    def type_check(self, embedding):
        if isinstance(embedding, list):
            embedding = torch.stack(embedding)
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        return embedding
    
    def __call__(self, x_embedding, y_embedding):
        return self.similarity_metric(x_embedding, y_embedding)

class CosineSimilarity(SimilarityMetrics):
    def __init__(self):
        super().__init__(nn.CosineSimilarity(dim=0))
    
    def __call__(self, x_embedding, y_embedding):
        x_embedding = self.type_check(x_embedding)
        y_embedding = self.type_check(y_embedding)
        
        if x_embedding.ndim == 1:
            x_embedding = x_embedding.view(-1, 1)
        if y_embedding.ndim == 1:
            y_embedding = y_embedding.view(-1, 1)
        return super().__call__(x_embedding, y_embedding)

class EucledianDistance(SimilarityMetrics):
    def __init__(self):
        super().__init__(nn.PairwiseDistance(p=2))
    
    def __call__(self, x_embedding, y_embedding):
        x_embedding = self.type_check(x_embedding)
        y_embedding = self.type_check(y_embedding)
        
        if x_embedding.ndim == 1:
            x_embedding = x_embedding.unsqueeze(0)
        if y_embedding.ndim == 1:
            y_embedding = y_embedding.unsqueeze(0)
        return -super().__call__(x_embedding, y_embedding)

class JaccardSimilarity(SimilarityMetrics):
    def __init__(self):
        pass

    def __call__(self, x_embedding, y_embedding):
        x_embedding = self.type_check(x_embedding)
        y_embedding = self.type_check(y_embedding)

        intersection = torch.sum(torch.min(x_embedding, y_embedding))
        union = torch.sum(torch.max(x_embedding, y_embedding))

        jaccard_similarity = intersection / (union + 1e-8)

        return jaccard_similarity
