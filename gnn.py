import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import to_undirected, to_dense_adj, dense_to_sparse
from torch_geometric.nn.pool import knn_graph
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
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
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, outliers_fraction=0.1):
        # build a PyG GAE
        self.encoder = GAEEncoder(input_dim, hidden_dim, num_layers)
        self.model = GAE(self.encoder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.outliers_fraction = outliers_fraction

    def _create_graph(self, X):
        """Create a graph from the input data.
        Args:
            X: Input data tensor of shape [batch_size, num_features]
        Returns:
            Data object with x and edge_index
        """
        # Ensure X is 2D and float
        if len(X.shape) == 1:
            X = X.unsqueeze(0)  # Add batch dimension if missing

        # Convert to float if needed and move to device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        elif X.dtype != torch.float:
            X = X.float()
        X = X.to(self.device)

        # Create a fully connected graph with self-loops
        num_nodes = X.size(0)
        if num_nodes == 0:
            raise ValueError("Input tensor has 0 nodes")

        # Create edge_index for a fully connected graph
        rows = []
        cols = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                rows.append(i)
                cols.append(j)

        # Create edge_index as long tensor (required by GCNConv)
        edge_index = torch.tensor([rows, cols], dtype=torch.long, device=self.device)

        # Create PyG Data object
        return Data(x=X, edge_index=edge_index)

    def fit(self, X, epochs=100, lr=1e-3, set_size=1000):
        """Train the GNN model.
        Args:
            X: Input data tensor
            epochs: Number of epochs to train
            lr: Learning rate
            set_size: Number of samples to use in each training iteration
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        X = X.to(self.device)

        # Calculate total number of iterations
        n_samples = len(X)
        n_sets = (n_samples + set_size - 1) // set_size  # ceiling division
        total_iterations = epochs * n_sets

        pbar = tqdm(range(total_iterations), desc="Training GNN", disable=not wandb.run)
        iteration = 0

        for epoch in range(epochs):
            # Shuffle indices at the start of each epoch
            indices = torch.randperm(n_samples)

            for set_start in range(0, n_samples, set_size):
                # Get a subset of data
                set_indices = indices[set_start : set_start + set_size]
                set_X = X[set_indices]

                # Create graph for this set
                data = self._create_graph(set_X)

                optimizer.zero_grad()
                # encode → z: [set_size, hidden_dim]
                z = self.model.encode(data.x, data.edge_index)
                # loss = edge-reconstruction loss
                loss = self.model.recon_loss(z, data.edge_index)
                loss.backward()
                optimizer.step()

                # Update progress bar
                iteration += 1
                pbar.set_description(
                    f"Epoch {epoch+1}/{epochs} Set {set_start//set_size + 1}/{n_sets}"
                )
                pbar.set_postfix({"recon_loss": f"{loss.item():.4f}"})

                # Log to wandb if enabled
                if wandb.run is not None:
                    wandb.log(
                        {
                            "gnn/epoch": epoch + 1,
                            "gnn/set": set_start // set_size + 1,
                            "gnn/iteration": iteration,
                            "gnn/recon_loss": loss.item(),
                        }
                    )

                pbar.update(1)

        pbar.close()
        return loss.item(), self

    def decision_function(self, X):
        """
        Returns one score per node: its MSE between
        reconstructed adjacency row and true adjacency row.
        """
        self.model.eval()
        with torch.no_grad():
            data = self._create_graph(X)
            z = self.model.encode(data.x, data.edge_index)
            # reconstruct full adjacency
            adj_rec = torch.sigmoid(z @ z.t())  # [N,N]
            adj_true = to_dense_adj(data.edge_index)[0].to(adj_rec)  # [N,N]
            # per-node MSE
            errors = ((adj_rec - adj_true) ** 2).mean(dim=1)
        return errors.cpu().numpy()

    def predict(self, X, percentile=95):
        """
        Label as anomaly (–1) any node whose
        recon-error is above the given percentile.
        """
        scores = self.decision_function(X)
        thresh = np.percentile(scores, percentile)
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
