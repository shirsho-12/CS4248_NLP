
# TODO: Add imports here

class Metrics:
    def __init__(self):
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

    def accuracy(self):
        # TODO: Add functionality
        self.metrics['accuracy'] = None
        return self.metrics['accuracy']

    def precision_recall_f1(self):
        # TODO: Add functionality
        self.metrics['precision'] = None
        self.metrics['recall'] = None
        self.metrics['f1'] = None
        return self.metrics['precision'], self.metrics['recall'], self.metrics['f1']

    def roc_auc(self):
        # TODO: Add functionality
        self.metrics['roc_auc'] = None
        return self.metrics['roc_auc']

    def mrr(self):
        # TODO: Add functionality
        self.metrics['mrr'] = None
        return self.metrics['mrr']

    def precision_at_k(self):
        # TODO: Add functionality
        self.metrics['precision_at_k'] = None
        return self.metrics['precision_at_k']
    
    def hit_rate(self):
        # TODO: Add functionality
        self.metrics['hit_rate'] = None
        return self.metrics['hit_rate']

    def confusion_matrix(self):
        # TODO: Add functionality
        self.metrics['confusion_matrix'] = None
        return self.metrics['confusion_matrix']

    def plot_roc_auc(self):
        # TODO: Add plot
        pass

    def plot_confusion_matrix(self):
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
