import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import to_undirected, to_dense_adj, dense_to_sparse
from torch_geometric.nn.pool import knn_graph
from torch_geometric.data import Data
import numpy as np
from tqdm.auto import tqdm
import wandb


class GAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        # first layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        # remaining hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # final embedding layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        # no activation on last layer
        return self.convs[-1](x, edge_index)


class GAEBenchmark:
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=3,
        outliers_fraction=0.1,
        lr=1e-3,
        epochs=100,
    ):
        # build a PyG GAE
        self.encoder = GAEEncoder(input_dim, hidden_dim, num_layers)
        self.model = GAE(self.encoder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.outliers_fraction = outliers_fraction
        self.lr = lr
        self.epochs = epochs

    def _create_graph(self, data):
        """Convert data points to a graph structure using k-nearest neighbors.
        Args:
            data: Input data as numpy array or torch.Tensor of shape [N, F]
        Returns:
            Data object with x and edge_index
        """
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Convert to tensor and move to device
        x = torch.tensor(data, dtype=torch.float, device=self.device)

        # Create k-nearest neighbors graph
        k = min(5, len(data) - 1)  # Use k=5 or less if dataset is small
        edge_index = knn_graph(x, k=k, loop=False)
        edge_index = to_undirected(edge_index)  # Ensure undirected edges

        return Data(x=x, edge_index=edge_index)

    def fit(self, X):
        """Train the GNN model.
        Args:
            X: Input data tensor (dataset.data)
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Create graph for the whole dataset
        data = self._create_graph(X)

        # Training loop
        for epoch in tqdm(range(self.epochs), desc="Training GNN"):
            optimizer.zero_grad()
            # encode → z: [n_samples, hidden_dim]
            z = self.model.encode(data.x, data.edge_index)
            # loss = edge-reconstruction loss
            loss = self.model.recon_loss(z, data.edge_index)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log to wandb if enabled
            if wandb.run is not None:
                wandb.log(
                    {
                        "gnn/epoch": epoch,
                        "gnn/recon_loss": loss.item(),
                        "gnn/learning_rate": scheduler.get_last_lr()[0],
                    }
                )

        return loss.item(), self

    def decision_function(self, X):
        """
        Returns one score per node: its MSE between
        reconstructed adjacency row and true adjacency row.
        Higher scores indicate anomalies.
        """
        self.model.eval()
        with torch.no_grad():
            data = self._create_graph(X)
            z = self.model.encode(data.x, data.edge_index)
            # reconstruct full adjacency
            adj_rec = torch.sigmoid(z @ z.t())  # [N,N]
            adj_true = to_dense_adj(data.edge_index)[0].to(adj_rec)  # [N,N]
            # per-node MSE (higher values indicate anomalies)
            errors = ((adj_rec - adj_true) ** 2).mean(dim=1)
            # Normalize scores to [0, 1] range
            errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
        return errors.cpu().numpy()

    def predict(self, X):
        """
        Label as anomaly (–1) any node whose
        recon-error is above the given percentile.
        """
        scores = self.decision_function(X)
        thresh = np.percentile(scores, 100 - 100 * self.outliers_fraction)
        return np.where(scores > thresh, -1, 1)


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GraphAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.encoder_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.decoder_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.decoder_layers.append(GCNConv(hidden_dim, input_dim))

    def forward(self, x, edge_index):
        # Encoder
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h, edge_index))

        # Decoder
        for layer in self.decoder_layers:
            h = F.relu(layer(h, edge_index))

        return h

    def fit(self, dataset, epochs, lr, batch_size, device, verbose=True):
        """Train the GNN model."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Convert data to PyTorch tensors
        X = torch.FloatTensor(dataset.data).to(device)
        edge_index, _ = dense_to_sparse(torch.ones((X.shape[0], X.shape[0])))
        edge_index = edge_index.to(device)

        # Training loop with progress bar
        pbar = tqdm(range(epochs), desc="Training GNN", disable=not verbose)
        for epoch in pbar:
            optimizer.zero_grad()
            output = self(X, edge_index)
            loss = F.mse_loss(output, X)
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to wandb if available
            if hasattr(self, "wandb") and self.wandb is not None:
                self.wandb.log({"epoch": epoch, "reconstruction_loss": loss.item()})

    def predict(self, dataset, device):
        """Predict anomalies using reconstruction error."""
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(dataset.data).to(device)
            edge_index, _ = dense_to_sparse(torch.ones((X.shape[0], X.shape[0])))
            edge_index = edge_index.to(device)
            output = self(X, edge_index)
            reconstruction_error = F.mse_loss(output, X, reduction="none").mean(dim=1)
            return reconstruction_error.cpu().numpy()
