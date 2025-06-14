from math import log
import os
import pickle
import json
import time

import numpy as np
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dose import DoSE_SVM, kl_divergence


## scikit-learn models
def train_sklearn(epoch, dataset, model):
    train_loss = 0
    # fit the data and tag outliers
    model.fit(dataset.data)
    train_scores = model.decision_function(
        dataset.data
    )  # positive distances for inlier, negative for outlier
    train_loss = -1 * np.average(train_scores)  # reverse signage
    return train_loss, model


def validate_sklearn(epoch, dataset, model):
    # fit the data and tag outliers
    val_scores = model.decision_function(
        dataset.data
    )  # positive distances for inlier, negative for outlier
    val_loss = -1 * np.average(val_scores)  # reverse signage
    # For IsolationForest, negate scores so higher values indicate anomalies
    if type(model) == IsolationForest:
        val_scores = -val_scores
    val_auroc = roc_auc_score(dataset.labels, val_scores)
    return val_loss, val_auroc


def test_sklearn(seed, args, train_dataset, test_dataset):
    """Test sklearn models and return metrics."""
    start_time = time.time()

    # Load model
    filename = os.path.join("results", f"{args.dataset}_{args.benchmark}_{seed}.pth")
    model = pickle.load(open(filename, "rb"))

    # Get predictions and scores
    scores = model.decision_function(test_dataset.data)
    predictions = model.predict(test_dataset.data)

    # Convert predictions to binary (0 for normal, 1 for anomaly)
    if type(model) == IsolationForest or type(model) == SGDOneClassSVM:
        predictions = [
            0 if y == 1 else 1 for y in predictions
        ]  # iForest sets 1 as inlier and -1 as outlier
    else:
        predictions = [0 if y == -1 else y for y in predictions]

    # For IsolationForest, negate scores so higher values indicate anomalies
    if type(model) == IsolationForest:
        scores = -scores

    # Calculate metrics
    accuracy = accuracy_score(test_dataset.labels, predictions)
    precision = precision_score(test_dataset.labels, predictions)
    recall = recall_score(test_dataset.labels, predictions)
    f1 = f1_score(test_dataset.labels, predictions)
    auroc = roc_auc_score(test_dataset.labels, scores)

    # Print results for this seed
    print(f"\nResults for seed {seed}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    # Save results for this seed
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "time": time.time() - start_time,
    }

    return results


## VAE + DoSE(SVM)
def train_vae(epoch, data_loader, model, prior, optimiser, device):
    model.train()
    zs = []
    train_loss = 0
    for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
        x, y = x.to(device=device, non_blocking=True), y.to(
            device=device, non_blocking=True
        )
        observation, posterior, z = model(x)
        loss = -observation.log_prob(x) + kl_divergence(z, posterior, prior)
        loss = -torch.logsumexp(-loss.view(loss.size(0), -1), dim=1).mean() - log(1)
        zs.append(z.detach())  # Store posterior samples
        train_loss += loss.item()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return train_loss / len(data_loader), torch.cat(zs)


def validate_vae(epoch, data_loader, model, prior, device):
    model.eval()
    val_loss = 0
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
            x, y = x.to(device=device, non_blocking=True), y.to(
                device=device, non_blocking=True
            )
            observation, posterior, z = model(x)
            loss = -observation.log_prob(x) + kl_divergence(z, posterior, prior)
            val_loss += -torch.logsumexp(
                -loss.view(loss.size(0), -1), dim=1
            ).mean() - log(1)
            # Store scores and labels for AUROC
            all_scores.extend(-loss.view(loss.size(0), -1).mean(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_auroc = roc_auc_score(all_labels, all_scores)
    return val_loss.item() / len(data_loader), val_auroc


def test_vae(seed, args, train_dataset, test_dataset):
    # Calculate result over ensemble of trained models
    # Load dataset summary statistics
    train_summary_stats = torch.load(
        os.path.join(
            "stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_train.pth"
        )
    )
    val_summary_stats = torch.load(
        os.path.join(
            "stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_val.pth"
        )
    )
    test_summary_stats = torch.load(
        os.path.join(
            "stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_test.pth"
        )
    )
    print(f"train shape: {train_summary_stats.shape}")
    print(f"test shape: {test_summary_stats.shape}")

    dose_svm = DoSE_SVM(train_summary_stats)
    outlier_preds = dose_svm.detect_outliers(test_summary_stats)
    return outlier_preds


def get_marginal_posterior(data_loader, model, device):
    model.eval()
    posteriors = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
            x, y = x.to(device=device, non_blocking=True), y.to(
                device=device, non_blocking=True
            )
            posteriors.append(model.encode(x))
    means, stddevs = torch.cat([p.mean for p in posteriors], dim=0), torch.cat(
        [p.stddev for p in posteriors], dim=0
    )
    mix = Categorical(torch.ones(means.size(0), device=device))
    comp = Independent(Normal(means, stddevs), 1)
    return MixtureSameFamily(mix, comp)


def train_gnn(epoch, dataset, model, set_size=1000):
    """Train GNN model for one epoch."""
    train_loss, model = model.fit(dataset.data, set_size=set_size)
    return train_loss, model


def validate_gnn(epoch, dataset, model):
    """Validate GNN model."""
    val_scores = model.decision_function(dataset.data)
    val_loss = -1 * np.average(val_scores)  # reverse signage to match other models
    val_auroc = roc_auc_score(dataset.labels, val_scores)
    return val_loss, val_auroc


def test_gnn(seed, args, train_dataset, test_dataset):
    """Test GNN model and return metrics."""
    start_time = time.time()

    # Load model
    filename = os.path.join("results", f"{args.dataset}_{args.benchmark}_{seed}.pth")
    model = pickle.load(open(filename, "rb"))

    # Get predictions and scores
    scores = model.decision_function(test_dataset.data)
    predictions = model.predict(test_dataset.data)

    # Convert predictions to binary (0 for normal, 1 for anomaly)
    predictions = [
        1 if y == -1 else 0 for y in predictions
    ]  # -1 is anomaly, 1 is normal

    # Calculate metrics
    accuracy = accuracy_score(test_dataset.labels, predictions)
    precision = precision_score(test_dataset.labels, predictions)
    recall = recall_score(test_dataset.labels, predictions)
    f1 = f1_score(test_dataset.labels, predictions)
    auroc = roc_auc_score(test_dataset.labels, scores)

    # Print results for this seed
    print(f"\nResults for seed {seed}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    # Save results for this seed
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "time": time.time() - start_time,
    }

    return results
