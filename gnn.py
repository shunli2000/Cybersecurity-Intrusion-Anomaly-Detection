import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn.pool import knn_graph
import numpy as np


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super(GNNModel, self).__init__()
        # Encoder layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Decoder layers for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Anomaly scoring layer
        self.anomaly_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        # Store intermediate representations
        node_embeddings = []

        # Encoder: Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            node_embeddings.append(x)

        # Get final node embeddings
        final_embeddings = node_embeddings[-1]

        # Decoder: Reconstruct input features
        reconstructed = self.decoder(final_embeddings)

        # Anomaly scores
        anomaly_scores = self.anomaly_head(final_embeddings)

        return {
            "node_embeddings": final_embeddings,
            "reconstructed": reconstructed,
            "anomaly_scores": anomaly_scores,
        }


class GNNBenchmark:
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        self.model = GNNModel(input_dim, hidden_dim, num_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _create_graph(self, data):
        """Convert data points to a graph structure using k-nearest neighbors."""
        # Handle tensor conversion properly
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Convert to tensor and move to device
        x = torch.tensor(data, dtype=torch.float, device=self.device)

        # Create k-nearest neighbors graph directly using PyG's knn_graph
        k = min(5, len(data) - 1)  # Use k=5 or less if dataset is small
        edge_index = knn_graph(x, k=k, loop=False)

        # Ensure undirected edges
        edge_index = to_undirected(edge_index)

        return Data(x=x, edge_index=edge_index)

    def fit(self, X):
        """Train the GNN model."""
        print("GNN: Starting fit method")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Create graph from data
        print("GNN: Creating graph from data")
        data = self._create_graph(X)
        print(
            f"GNN: Graph created with {data.num_nodes} nodes and {data.num_edges} edges"
        )

        # Training loop
        total_loss = 0
        print("GNN: Starting training loop")
        for epoch in range(100):  # 100 epochs
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data.x, data.edge_index)

            # Calculate reconstruction loss (MSE between input features and reconstructed features)
            reconstruction_loss = F.mse_loss(outputs["reconstructed"], data.x)

            # Calculate anomaly score loss (encourage normal samples to have low scores)
            anomaly_loss = torch.mean(torch.abs(outputs["anomaly_scores"]))

            # Combine losses
            loss = reconstruction_loss + 0.1 * anomaly_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch % 10 == 0:  # Print every 10 epochs
                print(
                    f"GNN: Epoch {epoch}, Total Loss: {loss.item():.4f}, "
                    f"Recon Loss: {reconstruction_loss.item():.4f}, "
                    f"Anomaly Loss: {anomaly_loss.item():.4f}"
                )

        avg_loss = total_loss / 100  # Average loss over epochs
        print(f"GNN: Training completed. Average loss: {avg_loss:.4f}")
        return avg_loss, self

    def decision_function(self, X):
        """Compute anomaly scores for each node."""
        self.model.eval()
        with torch.no_grad():
            data = self._create_graph(X)
            outputs = self.model(data.x, data.edge_index)
            return outputs["anomaly_scores"].cpu().numpy().flatten()

    def predict(self, X):
        """Predict normal (1) or anomaly (-1) for each node."""
        scores = self.decision_function(X)
        return np.where(scores > 0, 1, -1)  # Threshold at 0
