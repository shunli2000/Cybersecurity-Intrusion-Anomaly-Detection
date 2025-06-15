import argparse

import numpy as np
import torch

from benchmarks import BENCHMARK_LIST
from dataset import DATASETS

BENCHMARK_LIST = ["rcov", "svm", "ifor", "dose", "gnn"]


def configure():
    parser = argparse.ArgumentParser(description="Michelin Star Restaurant Process")
    # General flags
    parser.add_argument(
        "--dataset",
        type=str,
        default="beth",
        choices=list(DATASETS.keys()),
        metavar="D",
        help="Dataset selection",
    )
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        metavar="S",
        help="Factor by which to subsample the dataset (0 means no subsampling)",
    )
    parser.add_argument(
        "--vis-latents",
        action="store_true",
        help="True if want to visualise latent space",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="True if want to visualise dataset (and each epoch)",
    )
    # Training/Testing flags
    parser.add_argument("--test", action="store_true", help="Test benchmarks")
    parser.add_argument("--train", action="store_true", help="Train benchmarks")
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="B", help="Minibatch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="W",
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, metavar="N", help="Training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, metavar="P", help="Early stopping patience"
    )
    # Model flags
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ifor",
        choices=BENCHMARK_LIST,
        help="Override fitting of VAE model with specified benchmark",
    )
    parser.add_argument(
        "--outliers-fraction",
        type=float,
        default=0.1,
        help="Assumed proportion of the data that is an outlier",
    )  # used in rcov and ifor
    # # VAE
    parser.add_argument(
        "--latent-size", type=int, default=2, metavar="Z", help="Latent size"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, metavar="W", help="Weight decay"
    )
    # GNN
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, metavar="L", help="Learning rate"
    )
    parser.add_argument(
        "--gnn-epochs", type=int, default=100, help="Number of epochs for GNN training"
    )
    parser.add_argument(
        "--input-dim", type=int, default=None, help="Input dimension (for GNN)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, metavar="H", help="Hidden size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of layers in GNN"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for GNN")
    # Wandb configuration
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="islakong-carnegie-mellon-university",
        help="Wandb entity/username",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="BETH",
        help="Wandb project name",
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--wandb-tags", type=str, nargs="+", default=[], help="Tags for wandb run"
    )
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.disable_cuda
    args.device = torch.device("cuda" if use_cuda else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    return args
