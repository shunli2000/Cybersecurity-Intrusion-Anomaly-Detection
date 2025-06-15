import os
import pickle
import json
import time
import argparse
import wandb

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
from tqdm.auto import tqdm

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
    train_gnn,
    validate_gnn,
    test_gnn,
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
        f"Data loaded. Train shape: {train_dataset.data.shape}, "
        f"Val shape: {val_dataset.data.shape}, "
        f"Test shape: {test_dataset.data.shape}"
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
        use_vae = False
        if args.input_dim is None:
            args.input_dim = train_dataset.data.shape[1]
        model = get_benchmark(model_name, args)
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
    train_loss_log, val_loss_log, val_auroc_log = [], [], []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train model
        if use_vae:
            train_loss, zs = train_vae(
                epoch, train_loader, model, prior, optimiser, args.device
            )
        elif model_name == "gnn":
            train_loss, model = train_gnn(epoch, train_dataset, model)
        else:
            train_loss, model = train_sklearn(epoch, train_dataset, model)

        train_loss_log.append(train_loss)

        # Validate model
        if use_vae:
            val_loss, val_auroc = validate_vae(
                epoch, val_loader, model, prior, args.device
            )
        elif model_name == "gnn":
            val_loss, val_auroc = validate_gnn(epoch, val_dataset, model)
        else:
            val_loss, val_auroc = validate_sklearn(epoch, val_dataset, model)

        # Print metrics
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}"
        )

        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/auroc": val_auroc,
                }
            )

        # Save best model
        if len(val_loss_log) == 0 or val_loss < min(val_loss_log):
            filename = os.path.join(
                "results", f"{args.dataset}_{args.benchmark}_{args.seed}.pth"
            )
            pickle.dump(model, open(filename, "wb"))
            if args.use_wandb:
                wandb.save(filename)

        # Early stopping on validation loss
        if len(val_loss_log[-args.patience :]) >= args.patience and val_loss >= max(
            val_loss_log[-args.patience :]
        ):
            print(f"Early stopping at epoch {epoch}")
            break

        # Log validation metrics
        val_loss_log.append(val_loss)
        val_auroc_log.append(val_auroc)

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
            plot_line(
                range(1, epoch + 1),
                val_auroc_log,
                filename=f"{args.dataset}_{model_name}_auroc_val",
                xlabel="Epoch",
                ylabel="Validation AUROC",
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

    # Log final metrics to wandb
    if args.use_wandb:
        wandb.log(
            {
                "best_val_loss": min(val_loss_log),
                "best_val_auroc": max(val_auroc_log),
                "final_epoch": epoch,
            }
        )

    # results = {
    #     "train_loss": train_loss_log,
    #     "val_loss": val_loss_log,
    #     "val_auroc": val_auroc_log,
    #     "val_loss_min": min(val_loss_log),
    #     "val_auroc_max": max(val_auroc_log),
    # }

    # os.makedirs("results", exist_ok=True)
    # with open(f"results/{args.benchmark}_{args.seed}_train_results.json", "w") as f:
    #     json.dump(results, f, indent=4)

    return model


def test(args):
    """Test the model and output metrics."""
    print("\nTesting phase...")

    # Load datasets
    if args.dataset in DATASETS.keys():
        train_dataset, test_dataset = [
            DATASETS[args.dataset](split=split, subsample=args.subsample)
            for split in ["train", "test"]
        ]
    else:
        raise Exception("Invalid dataset specified")

    use_vae = True if args.benchmark == "dose" else False
    model_name = args.benchmark

    # Run test
    print(f"\nRun {args.dataset}_{args.benchmark}_1 at {datetime.now()}")

    if use_vae:
        results = test_vae(1, args, train_dataset, test_dataset)
    elif model_name == "gnn":
        results = test_gnn(1, args, train_dataset, test_dataset)
    else:
        results = test_sklearn(1, args, train_dataset, test_dataset)

    # Print results
    print(f"\nResults for {args.benchmark}:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"AUROC: {results['auroc']:.4f}")

    # Log results to wandb if enabled
    if args.use_wandb:
        wandb.log(
            {
                "test/accuracy": results["accuracy"],
                "test/precision": results["precision"],
                "test/recall": results["recall"],
                "test/f1": results["f1"],
                "test/auroc": results["auroc"],
            }
        )

        # Create a table with the results
        results_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Accuracy", results["accuracy"]],
                ["Precision", results["precision"]],
                ["Recall", results["recall"]],
                ["F1 Score", results["f1"]],
                ["AUROC", results["auroc"]],
            ],
        )
        wandb.log({"test/results": results_table})

    return results


def main():
    start = datetime.now()  # DEBUG
    print("Start: ", start)  # DEBUG
    os.makedirs("results", exist_ok=True)  # Where plots and pickled models are stored
    os.makedirs("stats", exist_ok=True)  # Where summary stats for DoSE are stored
    args = configure()

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "dataset": args.dataset,
                "benchmark": args.benchmark,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
            },
            name=args.wandb_name,
        )

    if args.train:
        train(args)

    if args.test:
        test(args)

    if args.use_wandb:
        wandb.finish()

    end = datetime.now()  # DEBUG
    print(f"Time to Complete: {end - start}")  # DEBUG


if __name__ == "__main__":
    main()
