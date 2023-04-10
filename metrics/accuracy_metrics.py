
# TODO: Add imports here

"""
Metrics class to calculate and store metrics for a given model
    @param rank_scores: A tuple of tuples containing the score according to the 
                        similarity metric and the actual rank of the emoji sequence
                        i.e. ((score, rank), (score, rank), ...)
    @param y: The concept of emoji sequence
"""
class AccuracyMetrics:
    def __init__(self, k=5):
        self.metrics = {}

    """
    List of metrics:
    - Accuracy
    - Precision/Recall and F1 + Confusion Matrix
    - ROC/AUC Curve
    - Mean Reciprocal Rank (MRR)
    - Precision@k
    - Hit Rate
    """

    def accuracy(self, rank_scores):
        correct = 0
        for i in range(len(rank_scores)):
            if rank_scores[i][1] == i:
                correct += 1
        self.metrics['accuracy'] = correct / len(rank_scores)
        return self.metrics['accuracy']
    
    def top_k_accuracy(self, rank_scores):
        correct = 0
        k = self.k
        for i in range(k):
            if rank_scores[i][1] == i:
                correct += 1
        self.metrics['top_k_accuracy'] = correct / k
        return self.metrics['top_k_accuracy']

    def precision_recall_f1(self, rank_scores, y):
        # TODO: Add functionality
        self.metrics['precision'] = None
        self.metrics['recall'] = None
        self.metrics['f1'] = None
        return self.metrics['precision'], self.metrics['recall'], self.metrics['f1']

    def roc_auc(self, rank_scores, y):
        # TODO: Add functionality
        self.metrics['roc_auc'] = None
        return self.metrics['roc_auc']

    def mrr(self, rank_scores, y):
        # TODO: Add functionality
        self.metrics['mrr'] = None
        return self.metrics['mrr']

    def precision_at_k(self, rank_scores, y):
        # TODO: Add functionality
        self.metrics['precision_at_k'] = None
        return self.metrics['precision_at_k']
    
    def hit_rate(self, rank_scores, y):
        # TODO: Add functionality
        self.metrics['hit_rate'] = None
        return self.metrics['hit_rate']

    def confusion_matrix(self, rank_scores, y):
        # TODO: Add functionality
        self.metrics['confusion_matrix'] = None
        return self.metrics['confusion_matrix']

    def plot_roc_auc(self, rank_scores, y):
        # TODO: Add plot
        pass

    def plot_confusion_matrix(self, rank_scores, y):
        # TODO: Add plot
        pass

    # Helper methods for metrics
    def get(self, name):
        return self.metrics[name]

    def get_all(self):
        return self.metrics

    def __str__(self):
        return str(self.metrics)

    def __repr__(self):
        return str(self.metrics)

    def __iter__(self):
        return iter(self.metrics)
