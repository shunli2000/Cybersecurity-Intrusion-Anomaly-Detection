import numpy as np
import os
import seaborn as sns
from sklearn.decomposition import PCA
from gnn import GAEBenchmark

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
        self.pca = PCA(n_components=args.latent_size)

    def fit(self, X):
        self.pca = self.pca.fit(X)
        X_whitened = self.pca.transform(X)
        self.base_model.fit(X_whitened)
        return self

    def decision_function(self, X):
        X_whitened = self.pca.transform(X)
        return self.base_model.decision_function(X_whitened)

    def predict(self, X):
        X_whitened = self.pca.transform(X)
        return self.base_model.predict(X_whitened)


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
