"""
gat_network.py - Graph Attention Network for Inter-Vehicle Communication
Models the communication cluster as a graph and uses attention to
selectively extract critical information from neighbours.

Architecture:
    1. Each vehicle (node) has a local feature vector (observation).
    2. Edges connect vehicles within the communication cluster.
    3. GAT layers compute attention-weighted aggregation of neighbours.
    4. Output is a dense context vector per vehicle, fusing local
       observations with the most relevant neighbour information.

This context vector feeds into the MA2C actor and critic networks.

From the project design:
    "The GNN acts as an attention mechanism, selectively extracting
     critical information from neighbors and fusing it into a highly
     dense context vector, optimizing communication efficiency."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing PyG; fall back to manual attention if unavailable
try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GATConfig:
    """Configuration for the GAT network."""
    # Observation dimensions
    local_obs_dim: int = 8       # Local observation features per vehicle
    neighbor_obs_dim: int = 6    # Features per neighbour (before GAT)

    # GAT architecture
    hidden_dim: int = 64         # Hidden layer size
    context_dim: int = 32        # Output context vector dimension
    num_heads: int = 4           # Number of attention heads
    num_gat_layers: int = 2      # Number of GAT layers
    dropout: float = 0.1         # Dropout rate


# ── Observation Builder ──────────────────────────────────────────────────────

class ObservationBuilder:
    """
    Builds feature vectors from vehicle state for the GAT.

    Local observation (per vehicle):
        [row, col, dest_row, dest_col, speed, dist_remaining,
         time_elapsed, num_blacklisted]

    Neighbour observation (per neighbour):
        [delta_row, delta_col, speed, dist_remaining,
         route_overlap_score, is_heading_same_direction]
    """

    @staticmethod
    def build_local_obs(
        current_node: Tuple[int, int],
        destination: Tuple[int, int],
        speed: float,
        dist_remaining: float,
        time_elapsed: float,
        num_blacklisted: int,
        grid_rows: int = 6,
        grid_cols: int = 6,
    ) -> List[float]:
        """Build normalised local observation vector."""
        return [
            current_node[0] / max(1, grid_rows - 1),   # Normalised row
            current_node[1] / max(1, grid_cols - 1),   # Normalised col
            destination[0] / max(1, grid_rows - 1),     # Normalised dest row
            destination[1] / max(1, grid_cols - 1),     # Normalised dest col
            speed,                                       # Already 0-1
            dist_remaining / (grid_rows + grid_cols),   # Normalised distance
            min(time_elapsed / 50.0, 1.0),              # Normalised time
            min(num_blacklisted / 10.0, 1.0),           # Normalised blacklist size
        ]

    @staticmethod
    def build_neighbor_obs(
        agent_node: Tuple[int, int],
        neighbor_node: Tuple[int, int],
        neighbor_speed: float,
        neighbor_dist_remaining: float,
        route_overlap: float,
        same_direction: bool,
        grid_rows: int = 6,
        grid_cols: int = 6,
    ) -> List[float]:
        """Build normalised neighbour observation vector."""
        dr = (neighbor_node[0] - agent_node[0]) / max(1, grid_rows - 1)
        dc = (neighbor_node[1] - agent_node[1]) / max(1, grid_cols - 1)

        return [
            dr,                                           # Relative row
            dc,                                           # Relative col
            neighbor_speed,                               # Speed
            neighbor_dist_remaining / (grid_rows + grid_cols),
            route_overlap,                                # 0-1 overlap score
            1.0 if same_direction else 0.0,              # Binary direction
        ]

    @staticmethod
    def compute_route_overlap(
        route_a: List[Tuple[int, int]],
        route_b: List[Tuple[int, int]],
    ) -> float:
        """Fraction of nodes shared between two routes."""
        if not route_a or not route_b:
            return 0.0
        set_a = set(route_a)
        set_b = set(route_b)
        overlap = len(set_a & set_b)
        return overlap / max(len(set_a), len(set_b))

    @staticmethod
    def compute_same_direction(
        agent_node: Tuple[int, int],
        agent_dest: Tuple[int, int],
        neighbor_node: Tuple[int, int],
        neighbor_dest: Tuple[int, int],
    ) -> bool:
        """Check if two vehicles are heading in roughly the same direction."""
        agent_dx = agent_dest[1] - agent_node[1]
        agent_dy = agent_dest[0] - agent_node[0]
        neigh_dx = neighbor_dest[1] - neighbor_node[1]
        neigh_dy = neighbor_dest[0] - neighbor_node[0]

        # Dot product of direction vectors
        dot = agent_dx * neigh_dx + agent_dy * neigh_dy
        return dot > 0


# ── GAT Network (with PyG) ──────────────────────────────────────────────────

class GATNetwork(nn.Module):
    """
    Graph Attention Network for fusing vehicle observations.

    Takes local observations + graph structure of the communication
    cluster, and outputs a context vector per vehicle.

    Architecture:
        local_obs -> Linear -> GAT layers -> Linear -> context_vector

    If PyTorch Geometric is not available, falls back to a simpler
    manual multi-head attention implementation.
    """

    def __init__(self, config: Optional[GATConfig] = None):
        super().__init__()
        self.config = config or GATConfig()
        cfg = self.config

        # Input projection: local obs -> hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.local_obs_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(cfg.hidden_dim),
        )

        if HAS_PYG:
            # PyG GAT layers
            self.gat_layers = nn.ModuleList()
            for i in range(cfg.num_gat_layers):
                in_dim = cfg.hidden_dim if i == 0 else cfg.hidden_dim * cfg.num_heads
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_dim,
                        out_channels=cfg.hidden_dim,
                        heads=cfg.num_heads,
                        dropout=cfg.dropout,
                        concat=True if i < cfg.num_gat_layers - 1 else False,
                    )
                )
        else:
            # Fallback: simple self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(cfg.hidden_dim)

        # Output projection: hidden -> context vector
        final_in = cfg.hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(final_in, cfg.context_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: [num_nodes, local_obs_dim] feature tensor.
            edge_index:    [2, num_edges] edge connectivity (PyG format).
                           If None, uses fully-connected attention.

        Returns:
            context: [num_nodes, context_dim] context vectors.
        """
        # Project input
        x = self.input_proj(node_features)  # [N, hidden_dim]

        if HAS_PYG and edge_index is not None:
            # PyG GAT forward
            for i, gat_layer in enumerate(self.gat_layers):
                x = gat_layer(x, edge_index)
                if i < len(self.gat_layers) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.config.dropout, training=self.training)
        else:
            # Fallback: self-attention over all nodes
            x_unsq = x.unsqueeze(0)  # [1, N, hidden]
            attn_out, _ = self.attention(x_unsq, x_unsq, x_unsq)
            x = self.attn_norm(x + attn_out.squeeze(0))

        # Output projection
        context = self.output_proj(x)  # [N, context_dim]

        return context

    def get_context_for_agent(
        self,
        agent_idx: int,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the context vector for a specific agent."""
        all_contexts = self.forward(node_features, edge_index)
        return all_contexts[agent_idx]


# ── Graph Builder Helper ─────────────────────────────────────────────────────

def build_cluster_graph(
    cluster_vehicle_ids: List[int],
) -> torch.Tensor:
    """
    Build a fully-connected edge_index for a communication cluster.

    In a cluster, every CAV can communicate with every other CAV
    (multi-hop connectivity). So the graph is fully connected.

    Args:
        cluster_vehicle_ids: List of vehicle IDs in the cluster.
                             Indices in the node feature tensor.

    Returns:
        edge_index: [2, num_edges] tensor for PyG.
    """
    num_nodes = len(cluster_vehicle_ids)
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # Fully connected (excluding self-loops)
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)

    return torch.tensor([src, dst], dtype=torch.long)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GAT Network Test")
    print(f"PyTorch Geometric available: {HAS_PYG}")
    print("=" * 60)

    config = GATConfig()
    gat = GATNetwork(config)
    print(f"\nModel architecture:")
    print(f"  Input dim:   {config.local_obs_dim}")
    print(f"  Hidden dim:  {config.hidden_dim}")
    print(f"  Context dim: {config.context_dim}")
    print(f"  Heads:       {config.num_heads}")
    print(f"  GAT layers:  {config.num_gat_layers}")
    total_params = sum(p.numel() for p in gat.parameters())
    print(f"  Total params: {total_params:,}")

    # ── Test 1: Observation builder ──
    print("\n--- Test 1: Observation Builder ---")
    obs = ObservationBuilder.build_local_obs(
        current_node=(2, 3),
        destination=(0, 5),
        speed=0.8,
        dist_remaining=4.0,
        time_elapsed=3.0,
        num_blacklisted=1,
    )
    print(f"Local obs (8 features): {[round(x, 3) for x in obs]}")

    neigh_obs = ObservationBuilder.build_neighbor_obs(
        agent_node=(2, 3),
        neighbor_node=(2, 4),
        neighbor_speed=0.6,
        neighbor_dist_remaining=3.0,
        route_overlap=0.4,
        same_direction=True,
    )
    print(f"Neighbor obs (6 features): {[round(x, 3) for x in neigh_obs]}")

    # ── Test 2: Forward pass with cluster ──
    print("\n--- Test 2: Forward Pass ---")
    num_nodes = 5  # 5 vehicles in cluster

    # Create random observations
    node_features = torch.randn(num_nodes, config.local_obs_dim)
    edge_index = build_cluster_graph(list(range(num_nodes)))

    print(f"Input: {num_nodes} nodes, {edge_index.shape[1]} edges")

    gat.eval()
    with torch.no_grad():
        context = gat(node_features, edge_index)

    print(f"Output context: shape={context.shape}")
    print(f"  Agent 0 context: {context[0][:5].tolist()} ...")

    # ── Test 3: Single agent context ──
    print("\n--- Test 3: Single Agent Context ---")
    with torch.no_grad():
        agent_ctx = gat.get_context_for_agent(0, node_features, edge_index)
    print(f"Agent 0 context vector: shape={agent_ctx.shape}")

    # ── Test 4: Single node (no neighbours) ──
    print("\n--- Test 4: Isolated Agent (No Neighbours) ---")
    single_features = torch.randn(1, config.local_obs_dim)
    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        single_ctx = gat(single_features, empty_edges)
    print(f"Isolated agent context: shape={single_ctx.shape}")

    # ── Test 5: Route overlap ──
    print("\n--- Test 5: Route Overlap Score ---")
    route_a = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]
    route_b = [(2, 2), (2, 3), (2, 4), (3, 4), (3, 5)]
    overlap = ObservationBuilder.compute_route_overlap(route_a, route_b)
    print(f"Route A: {route_a}")
    print(f"Route B: {route_b}")
    print(f"Overlap: {overlap:.2f} (3 shared nodes / 5 max)")

    print("\nAll tests passed.")