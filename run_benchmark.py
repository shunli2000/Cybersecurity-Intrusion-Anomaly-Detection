import os
import pickle
import json
import time
import argparse

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch
from torch import optim
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import tqdm

from benchmarks import WhitenedBenchmark, get_benchmark
from config import configure, BENCHMARK_LIST
from dataset import BETHDataset, GaussianDataset, DATASETS
from vae import VAE
from dose import get_summary_stats
from plotting import plot_data, plot_line
from training import (
    get_marginal_posterior,
    test_sklearn,
    test_vae,
    train_sklearn,
    train_vae,
    validate_sklearn,
    validate_vae,
)

# DEBUG timestamps
from datetime import datetime  # DEBUG


####################################################
## Script
####################################################
def train(args):
    print("Starting data loading...")  # Debug print
    ##########################
    # Data
    ##########################
    train_dataset, val_dataset, test_dataset = [
        DATASETS[args.dataset](split=split, subsample=args.subsample)
        for split in ["train", "val", "test"]
    ]
    print(
        f"Data loaded. Train dataset shape: {train_dataset.data.shape}"
    )  # Debug print

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    if args.vis and hasattr(train_dataset, "plot"):
        plot_data(
            [train_dataset, val_dataset, test_dataset],
            ["train", "val", "test"],
            train_dataset,
            prefix=f"{args.dataset}_gaussian",
        )

    print("Starting model initialization...")  # Debug print
    ##########################
    # Model
    ##########################
    model_name = args.benchmark
    use_vae = True if model_name == "dose" else False

    if model_name == "dose":
        use_vae = True
        input_shape = train_dataset.get_input_shape()
        model = VAE(
            input_shape=input_shape,
            latent_size=args.latent_size,
            hidden_size=args.hidden_size,
            observation=train_dataset.get_distribution(),
        )
        model.to(device=args.device)
        optimiser = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        prior = MultivariateNormal(
            torch.zeros(args.latent_size, device=args.device),
            torch.eye(args.latent_size, device=args.device),
        )
    elif model_name == "gnn":
        print("Initializing GNN model...")  # Debug print
        use_vae = False
        if args.input_dim is None:
            args.input_dim = train_dataset.data.shape[1]
        print(f"Input dimension: {args.input_dim}")  # Debug print
        model = get_benchmark(model_name, args)
        print("GNN model initialized")  # Debug print
    else:
        use_vae = False
        if model_name == "rcov":
            model = EllipticEnvelope(contamination=args.outliers_fraction)
        elif model_name == "svm":
            base_model = SGDOneClassSVM(random_state=args.seed)
            model = WhitenedBenchmark(model_name, base_model, args)
        elif model_name == "ifor":
            base_model = IsolationForest(
                contamination=args.outliers_fraction, random_state=args.seed
            )
            model = WhitenedBenchmark(model_name, base_model, args)

    print("Starting training loop...")  # Debug print
    ##########################
    # Train & Validate
    ##########################
    train_loss_log, val_loss_log = [], []
    pbar = tqdm.trange(1, args.epochs + 1)
    for epoch in pbar:
        print(f"Starting epoch {epoch}")  # Debug print
        # Train model
        if use_vae:
            train_loss, zs = train_vae(
                epoch, train_loader, model, prior, optimiser, args.device
            )
        elif model_name == "gnn":
            print("Training GNN model...")  # Debug print
            train_loss, model = model.fit(train_dataset.data)
            print(f"GNN training completed. Loss: {train_loss}")  # Debug print
        else:
            train_loss, model = train_sklearn(epoch, train_dataset, model)
        pbar.set_description(f"Epoch: {epoch} | Train Loss: {train_loss}")
        train_loss_log.append(train_loss)

        # Validate model
        if use_vae:
            val_loss = validate_vae(epoch, val_loader, model, prior, args.device)
        elif model_name == "gnn":
            # GNN validation
            val_scores = model.decision_function(val_dataset.data)
            val_loss = -1 * np.average(
                val_scores
            )  # reverse signage to match other models
        else:
            val_loss = validate_sklearn(epoch, val_dataset, model)
        pbar.set_description(f"Epoch: {epoch} | Val Loss: {val_loss}")

        # Save best model
        if len(val_loss_log) == 0 or val_loss < min(val_loss_log):
            filename = os.path.join(
                "results", f"{args.dataset}_{args.benchmark}_{args.seed}.pth"
            )
            pickle.dump(model, open(filename, "wb"))
        # Early stopping on validation loss
        if len(val_loss_log[-args.patience :]) >= args.patience and val_loss >= max(
            val_loss_log[-args.patience :]
        ):
            print(f"Early stopping at epoch {epoch}")
            break
        val_loss_log.append(val_loss)

        # Plot losses
        if args.vis:
            plot_line(
                range(1, epoch + 1),
                train_loss_log,
                filename=f"{args.dataset}_{model_name}_loss_train",
                xlabel="Epoch",
                ylabel="Training Loss",
            )
            plot_line(
                range(1, epoch + 1),
                val_loss_log,
                filename=f"{args.dataset}_{model_name}_loss_val",
                xlabel="Epoch",
                ylabel="Validation Loss",
            )

        # Visualise model performance
        if args.vis:
            if use_vae:
                with torch.no_grad():
                    samples = model.decode(prior.sample((100,))).sample().cpu()
                    plot_data(
                        [train_dataset, samples],
                        ["training", model_name],
                        train_dataset,
                        prefix=f"{args.dataset}_{model_name}",
                        suffix=str(epoch),
                    )
                    if args.vis_latents and args.latent_size == 2:
                        prior_means = torch.stack([prior.mean])
                        plot_data(
                            [prior.sample((2000,)).cpu(), zs.cpu(), prior_means.cpu()],
                            ["Prior Samples", "Posterior Samples", "Prior Means"],
                            "",
                            prefix=f"{args.dataset}_gaussian_latents",
                            suffix=str(epoch),
                            xlim=[-8, 8],
                            ylim=[-8, 8],
                        )
            else:
                plot_data(
                    [train_dataset, model],
                    ["training", model_name],
                    train_dataset,
                    prefix=f"{args.dataset}_{model_name}",
                    suffix=str(epoch),
                )

    # Calculate summary statistics for DoSE later
    if use_vae:
        # Calculate marginal posterior distribution q(Z)
        marginal_posterior = get_marginal_posterior(train_loader, model, args.device)
        # Collect summary statistics of model on datasets
        train_summary_stats = get_summary_stats(
            train_loader, model, marginal_posterior, 16, 4, args.seed, args.device
        )
        val_summary_stats = get_summary_stats(
            val_loader, model, marginal_posterior, 16, 4, args.seed, args.device
        )
        test_summary_stats = get_summary_stats(
            test_loader, model, marginal_posterior, 16, 4, args.seed, args.device
        )
        # Save summary statistics
        torch.save(
            train_summary_stats,
            os.path.join(
                "stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_train.pth"
            ),
        )
        torch.save(
            val_summary_stats,
            os.path.join(
                "stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_val.pth"
            ),
        )
        torch.save(
            test_summary_stats,
            os.path.join(
                "stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_test.pth"
            ),
        )
    print(f"Min Val Loss: {min(val_loss_log)}")  # Print minimum validation loss


def test(args):
    if args.dataset in DATASETS.keys():
        train_dataset, test_dataset = [
            DATASETS[args.dataset](split=split, subsample=args.subsample)
            for split in ["train", "test"]
        ]
    else:
        raise Exception("Invalid dataset specified")

    use_vae = True if args.benchmark == "dose" else False
    outlier_preds = []
    for seed in tqdm.trange(1, 6):
        print(
            f"Run {args.dataset}_{args.benchmark}_{seed} at ", datetime.now()
        )  # DEBUG
        if use_vae:
            outlier_preds.append(test_vae(seed, args, train_dataset, test_dataset))
        else:
            outlier_preds.append(test_sklearn(seed, args, train_dataset, test_dataset))

    # Compare labels/predictions to labels
    outlier_preds = np.stack(outlier_preds, axis=0).mean(axis=0)
    print(
        f"Benchmark {args.benchmark} AUROC: {roc_auc_score(test_dataset.labels.numpy(), outlier_preds)}"
    )


def evaluate(model, test_dataset, args):
    """Evaluate the model on the test set."""
    print("\nEvaluating model...")
    start_time = time.time()

    # Get predictions
    scores = model.decision_function(test_dataset.data)
    predictions = model.predict(test_dataset.data)

    # Calculate metrics
    accuracy = accuracy_score(test_dataset.labels, predictions)
    precision = precision_score(test_dataset.labels, predictions)
    recall = recall_score(test_dataset.labels, predictions)
    f1 = f1_score(test_dataset.labels, predictions)
    auroc = roc_auc_score(test_dataset.labels, scores)

    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    # Save results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "time": time.time() - start_time,
    }

    # Save to file
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.benchmark}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def main():
    start = datetime.now()  # DEBUG
    print("Start: ", start)  # DEBUG
    os.makedirs("results", exist_ok=True)  # Where plots and pickled models are stored
    os.makedirs("stats", exist_ok=True)  # Where summary stats for DoSE are stored
    args = configure()

    if args.train:
        train(args)
    elif args.test:
        test(args)
    else:
        raise Exception("Must add flag --train or --test for benchmark functions")

    # Compare results if both models have been evaluated
    if os.path.exists("results/iforest_results.json") and os.path.exists(
        "results/gnn_results.json"
    ):
        print("\nComparing Results:")
        with open("results/iforest_results.json", "r") as f:
            iforest_results = json.load(f)
        with open("results/gnn_results.json", "r") as f:
            gnn_results = json.load(f)

        print("\nIsolationForest Results:")
        print(f"AUROC: {iforest_results['auroc']:.4f}")
        print(f"Accuracy: {iforest_results['accuracy']:.4f}")
        print(f"F1 Score: {iforest_results['f1']:.4f}")

        print("\nGNN Results:")
        print(f"AUROC: {gnn_results['auroc']:.4f}")
        print(f"Accuracy: {gnn_results['accuracy']:.4f}")
        print(f"F1 Score: {gnn_results['f1']:.4f}")

        print("\nImprovement:")
        print(f"AUROC: {gnn_results['auroc'] - iforest_results['auroc']:.4f}")
        print(f"Accuracy: {gnn_results['accuracy'] - iforest_results['accuracy']:.4f}")
        print(f"F1 Score: {gnn_results['f1'] - iforest_results['f1']:.4f}")

    end = datetime.now()  # DEBUG
    print(f"Time to Complete: {end - start}")  # DEBUG


if __name__ == "__main__":
    main()
