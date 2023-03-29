import torch 
from itertools import chain
from tqdm import tqdm

class UnsupervisedBaselineTrainer:
    def __init__(self, model, data_loader, similarity_metric, accuracy_metric):
        self.model = model
        self.data_loader = data_loader
        self.similarity_metric = similarity_metric
        self.accuracy_metric = accuracy_metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(self):
        """
        The baseline isn't supposed to be trained, so this method is empty.
        """
        pass

    def test(self):
        accuracy = 0
        for x, y in tqdm(self.data_loader):
            y = y[0]
            rank_scores = []
            y_score = self.model(y)
            for rank, sentences in x.items():
                sentences = list(chain(*sentences))
                embeddings = self.model(sentences).mean(axis=0) # simple mean
                rank_scores.append((self.similarity_metric(embeddings, y_score).cpu().detach().numpy()[0], rank))
            sorted_rank_scores = sorted(rank_scores, key=lambda x: x[0], reverse=True)
            accuracy += self.accuracy_metric(sorted_rank_scores)
        return accuracy / len(self.data_loader)
