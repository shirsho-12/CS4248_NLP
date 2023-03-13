import torch 

class UnsupervisedBaseline:
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
        for x, y in self.data_loader:
            y = y.to(self.device)
            rank_scores = ()
            for rank, sentences in x.items():
                sentences = sentences.to(self.device)
                embeddings = self.model(sentences)
                rank_scores += (self.similarity_metric(embeddings), rank)
            sorted_rank_scores = sorted(rank_scores, key=lambda x: x[0], reverse=True)
            accuracy += self.accuracy_metric(sorted_rank_scores, y)
        return accuracy / len(self.data_loader)


