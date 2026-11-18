"""Shallow MLP with optional training-fold class balancing."""
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


class CandidateMLP(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=.001,
                 learning_rate_init=.001, max_iter=150, random_state=42,
                 balance_classes=True):
        self.balance_classes = balance_classes
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                         learning_rate_init=learning_rate_init, max_iter=max_iter,
                         random_state=random_state, early_stopping=False,
                         n_iter_no_change=20)

    def fit(self, X, y, sample_weight=None):
        if self.balance_classes:
            weights = compute_sample_weight("balanced", y)
            sample_weight = weights if sample_weight is None else sample_weight * weights
        return super().fit(X, y, sample_weight=sample_weight)
