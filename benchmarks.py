import numpy as np
import os
import seaborn as sns
from sklearn.decomposition import PCA
from gnn import GAEBenchmark
from sklearn.ensemble import IsolationForest

BENCHMARK_LIST = ["rcov", "svm", "ifor", "dose"]

####################################################
## Custom Baseline Classes
####################################################


class WhitenedBenchmark:
    """
    Generic class to standardise scikit-learn model functions. Definitions for scikit-learn models available is in BENCHMARK.

    # TODO Approximation kernels: Nystroem, Random Fourier Features
    """

    def __init__(self, model_name, base_model, args):
        self.model_name = model_name
        self.base_model = base_model
        # Only use PCA for non-Isolation Forest models
        if not isinstance(base_model, IsolationForest):
            self.pca = PCA(n_components=args.latent_size)
        else:
            self.pca = None

    def fit(self, X):
        if self.pca is not None:
            self.pca = self.pca.fit(X)
            X_whitened = self.pca.transform(X)
            self.base_model.fit(X_whitened)
        else:
            # For Isolation Forest, use raw features
            self.base_model.fit(X)
        return self

    def decision_function(self, X):
        if self.pca is not None:
            X_whitened = self.pca.transform(X)
            return self.base_model.decision_function(X_whitened)
        else:
            # For Isolation Forest, use raw features
            return -self.base_model.score_samples(X)  # Negative scores for anomalies

    def predict(self, X):
        if self.pca is not None:
            X_whitened = self.pca.transform(X)
            return self.base_model.predict(X_whitened)
        else:
            # For Isolation Forest, use raw features
            return self.base_model.predict(X)


def get_benchmark(model_name, args):
    if model_name == "gnn":
        return GAEBenchmark(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.learning_rate,
            epochs=args.gnn_epochs,
            outliers_fraction=args.outliers_fraction,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
