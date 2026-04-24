"""
grid_network.py - Transportation Network for CAV Simulation
Builds the grid-based road network used by the Dec-CTDSP routing algorithm.

Based on the domain described in:
    Mostafizi et al., "A Decentralized and Coordinated Routing Algorithm
    for Connected and Autonomous Vehicles," IEEE Trans. ITS, 2022.

Network layout:
    - 6x6 grid of intersections (nodes labelled (row, col) where row=0 is top)
    - East-west streets: ONE-WAY towards east (left -> right)
    - North-south streets: TWO-WAY (up and down)
    - Origins: 5 nodes on the west edge  (rows 0-4, col 0)
    - Destinations: 5 nodes on the east edge (rows 0-4, col 5)
    - All links share the same speed limit and free-flow travel time
    - Speed-density relationship governs actual travel times under congestion
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



@dataclass
class NetworkConfig:
    """All tuneable parameters for the grid network."""
    rows: int = 6                    # Number of rows in grid
    cols: int = 6                    # Number of columns in grid
    block_length: float = 1.0        # Distance between adjacent intersections (in "blocks")
    speed_limit: float = 1.0         # Free-flow speed (blocks per timestep)
    free_flow_travel_time: float = 1.0  # Travel time on an empty link (timesteps)
    link_capacity: int = 5           # Max vehicles on a link before gridlock
    num_vehicles: int = 100          # Total vehicles in simulation



def speed_density(density: float, capacity: int, speed_limit: float = 1.0) -> float:
    """
    Compute the speed on a link given its current vehicle density.

    This replicates the speed-density curve from the Dec-CTDSP paper
    (Figure 2, bottom right). Speed decreases linearly from the speed
    limit as density approaches the link capacity.

    Args:
        density:     Number of vehicles currently on the link.
        capacity:    Maximum vehicles the link can hold.
        speed_limit: Speed when the link is empty.

    Returns:
        Current speed on the link (0.0 to speed_limit).
    """
    if density <= 0:
        return speed_limit
    if density >= capacity:
        return 0.0
    # Linear relationship: speed = speed_limit * (1 - density / capacity)
    return speed_limit * max(0.0, 1.0 - density / capacity)


def compute_travel_time(
    density: float,
    capacity: int,
    free_flow_tt: float = 1.0,
    speed_limit: float = 1.0
) -> float:
    """
    Compute travel time on a link given current density.

    Args:
        density:      Number of vehicles currently on the link.
        capacity:     Maximum vehicles the link can hold.
        free_flow_tt: Travel time when link is empty.
        speed_limit:  Free-flow speed.

    Returns:
        Current travel time. Returns a large value (100.0) if gridlocked.
    """
    spd = speed_density(density, capacity, speed_limit)
    if spd <= 1e-6:
        return 100.0  # Effectively gridlocked
    return free_flow_tt * (speed_limit / spd)



class GridNetwork:
    """
    The transportation network represented as a directed graph.

    Each node is an intersection identified by a (row, col) tuple.
    Each directed edge is a road link with attributes:
        - free_flow_tt:  travel time when empty
        - capacity:      max vehicles before gridlock
        - current_vehicles: list of vehicle IDs currently on this link
        - direction:     'east', 'north', or 'south'
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.origins: List[Tuple[int, int]] = []
        self.destinations: List[Tuple[int, int]] = []
        self._build_network()


    def _build_network(self) -> None:
        """Build the full grid graph with nodes, edges, origins, and destinations."""
        cfg = self.config

        # Create intersection nodes
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                self.graph.add_node(
                    (r, c),
                    pos=(c, cfg.rows - 1 - r),  # (x, y) for plotting
                    is_origin=(c == 0 and r < cfg.rows - 1),
                    is_destination=(c == cfg.cols - 1 and r < cfg.rows - 1),
                    is_blocked=False,
                )

        # East-West edges: ONE-WAY eastbound only
        for r in range(cfg.rows):
            for c in range(cfg.cols - 1):
                self._add_link((r, c), (r, c + 1), direction="east")

        # North-South edges: TWO-WAY
        for r in range(cfg.rows - 1):
            for c in range(cfg.cols):
                self._add_link((r, c), (r + 1, c), direction="south")
                self._add_link((r + 1, c), (r, c), direction="north")

        # Define origins (west edge, rows 0 through rows-2)
        # and destinations (east edge, rows 0 through rows-2)
        # Following the paper: 5 origins, 5 destinations
        self.origins = [(r, 0) for r in range(cfg.rows - 1)]
        self.destinations = [(r, cfg.cols - 1) for r in range(cfg.rows - 1)]

    def _add_link(
        self,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int],
        direction: str
    ) -> None:
        """Add a directed road link between two intersections."""
        cfg = self.config
        self.graph.add_edge(
            from_node,
            to_node,
            free_flow_tt=cfg.free_flow_travel_time,
            capacity=cfg.link_capacity,
            current_vehicles=[],       # Vehicle IDs on this link
            direction=direction,
            length=cfg.block_length,
        )


    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all nodes reachable from the given intersection."""
        return list(self.graph.successors(node))

    def get_edge_density(self, from_node: Tuple[int, int], to_node: Tuple[int, int]) -> int:
        """Return the number of vehicles currently on a link."""
        edge_data = self.graph.edges[from_node, to_node]
        return len(edge_data["current_vehicles"])

    def get_edge_travel_time(
        self,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int]
    ) -> float:
        """
        Compute the current travel time on a link based on density.

        This is the key function used by Dec-CTDSP's GetTravelTime.
        """
        edge_data = self.graph.edges[from_node, to_node]
        density = len(edge_data["current_vehicles"])
        return compute_travel_time(
            density=density,
            capacity=edge_data["capacity"],
            free_flow_tt=edge_data["free_flow_tt"],
            speed_limit=self.config.speed_limit,
        )

    def get_free_flow_travel_time(
        self,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int]
    ) -> float:
        """Return the free-flow travel time of a link (no congestion)."""
        return self.graph.edges[from_node, to_node]["free_flow_tt"]

    def is_node_blocked(self, node: Tuple[int, int]) -> bool:
        """Check if an intersection is currently blocked by an obstacle."""
        return self.graph.nodes[node].get("is_blocked", False)


    def block_node(self, node: Tuple[int, int]) -> None:
        """
        Mark an intersection as blocked (e.g., stalled car).

        Blocked nodes will be detected by vehicles and added to OMM blacklists.
        """
        if node in self.graph.nodes:
            self.graph.nodes[node]["is_blocked"] = True

    def unblock_node(self, node: Tuple[int, int]) -> None:
        """Remove a blockage from an intersection."""
        if node in self.graph.nodes:
            self.graph.nodes[node]["is_blocked"] = False

    def get_blocked_nodes(self) -> List[Tuple[int, int]]:
        """Return all currently blocked intersections."""
        return [
            n for n, data in self.graph.nodes(data=True)
            if data.get("is_blocked", False)
        ]


    def place_vehicle_on_link(
        self,
        vehicle_id: int,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int]
    ) -> bool:
        """
        Try to add a vehicle to a link's current vehicle list.

        Returns:
            True if placed or already present, False if link is full.
        """
        vehicles = self.graph.edges[from_node, to_node]["current_vehicles"]
        if vehicle_id in vehicles:
            return True

        capacity = self.graph.edges[from_node, to_node]["capacity"]
        if len(vehicles) >= capacity:
            return False

        vehicles.append(vehicle_id)
        return True

    def remove_vehicle_from_link(
        self,
        vehicle_id: int,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int]
    ) -> None:
        """Remove a vehicle from a link's current vehicle list."""
        vehicles = self.graph.edges[from_node, to_node]["current_vehicles"]
        if vehicle_id in vehicles:
            vehicles.remove(vehicle_id)


    def get_link_densities(self) -> Dict[Tuple, int]:
        """Return a dict of {(from, to): num_vehicles} for all links."""
        return {
            (u, v): len(data["current_vehicles"])
            for u, v, data in self.graph.edges(data=True)
        }

    def get_total_vehicles_on_network(self) -> int:
        """Count total vehicles currently on all links."""
        return sum(
            len(data["current_vehicles"])
            for _, _, data in self.graph.edges(data=True)
        )

    def get_network_summary(self) -> Dict:
        """Return a summary of the network topology."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_origins": len(self.origins),
            "num_destinations": len(self.destinations),
            "grid_size": f"{self.config.rows}x{self.config.cols}",
            "blocked_nodes": self.get_blocked_nodes(),
        }


    def get_node_position(self, node: Tuple[int, int]) -> Tuple[float, float]:
        """
        Get the (x, y) world position of a node.

        Used to compute Euclidean distances for communication radius.
        The position is in block units: col = x, inverted row = y.
        """
        return self.graph.nodes[node]["pos"]

    def euclidean_distance(
        self,
        node_a: Tuple[int, int],
        node_b: Tuple[int, int]
    ) -> float:
        """Compute Euclidean distance between two nodes in block units."""
        pos_a = self.get_node_position(node_a)
        pos_b = self.get_node_position(node_b)
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


    def visualize(
        self,
        title: str = "CAV Grid Transportation Network",
        show_densities: bool = False,
        blocked_nodes: Optional[List[Tuple[int, int]]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Draw the grid network using matplotlib.

        Args:
            title:          Plot title.
            show_densities: If True, colour edges by current vehicle count.
            blocked_nodes:  Highlight these nodes as blocked (red X).
            save_path:      If provided, save figure to this path instead of showing.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        pos = nx.get_node_attributes(self.graph, "pos")

        if show_densities:
            densities = [
                len(self.graph.edges[u, v]["current_vehicles"])
                for u, v in self.graph.edges()
            ]
            max_d = max(densities) if densities and max(densities) > 0 else 1
            edge_colors = [plt.cm.YlOrRd(d / max_d) for d in densities]
            edge_widths = [1.0 + 2.0 * (d / max_d) for d in densities]
        else:
            edge_colors = []
            edge_widths = []
            for u, v, data in self.graph.edges(data=True):
                if data["direction"] == "east":
                    edge_colors.append("#4A90D9")
                    edge_widths.append(1.8)
                else:
                    edge_colors.append("#7B8794")
                    edge_widths.append(1.2)

        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.05",
            alpha=0.7,
        )

        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node in self.origins:
                node_colors.append("#3B82F6")   # Blue for origins
                node_sizes.append(350)
            elif node in self.destinations:
                node_colors.append("#22C55E")   # Green for destinations
                node_sizes.append(350)
            else:
                node_colors.append("#E5E7EB")   # Light gray for intersections
                node_sizes.append(200)

        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="#374151",
            linewidths=1.0,
        )

        labels = {
            node: f"{node[0]},{node[1]}"
            for node in self.graph.nodes()
        }
        nx.draw_networkx_labels(
            self.graph, pos, labels, ax=ax,
            font_size=7, font_weight="bold",
        )

        if blocked_nodes:
            blocked_in_graph = [n for n in blocked_nodes if n in self.graph.nodes]
            if blocked_in_graph:
                blocked_pos = {n: pos[n] for n in blocked_in_graph}
                nx.draw_networkx_nodes(
                    self.graph, blocked_pos, nodelist=blocked_in_graph, ax=ax,
                    node_color="#EF4444",
                    node_size=300,
                    node_shape="X",
                    edgecolors="#991B1B",
                    linewidths=2.0,
                )

        legend_elements = [
            mpatches.Patch(facecolor="#3B82F6", edgecolor="#374151", label="Origins"),
            mpatches.Patch(facecolor="#22C55E", edgecolor="#374151", label="Destinations"),
            mpatches.Patch(facecolor="#E5E7EB", edgecolor="#374151", label="Intersections"),
        ]
        if blocked_nodes:
            legend_elements.append(
                mpatches.Patch(facecolor="#EF4444", edgecolor="#991B1B", label="Blocked")
            )
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        ax.annotate(
            "→ East (one-way)", xy=(0.5, -0.02), xycoords="axes fraction",
            ha="center", fontsize=9, color="#4A90D9", fontweight="bold",
        )
        ax.annotate(
            "↕ North-South (two-way)", xy=(1.02, 0.5), xycoords="axes fraction",
            ha="left", va="center", fontsize=9, color="#7B8794",
            fontweight="bold", rotation=90,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_aspect("equal")
        ax.margins(0.12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close(fig)



if __name__ == "__main__":
    # Build default 6x6 network
    config = NetworkConfig()
    network = GridNetwork(config)

    # Print summary
    summary = network.get_network_summary()
    print("=" * 50)
    print("Grid Network Summary")
    print("=" * 50)
    for key, val in summary.items():
        print(f"  {key}: {val}")

    # Test basic queries
    print(f"\nOrigins:      {network.origins}")
    print(f"Destinations: {network.destinations}")

    # Test neighbours of a central node
    test_node = (2, 3)
    neighbors = network.get_neighbors(test_node)
    print(f"\nNeighbors of {test_node}: {neighbors}")

    # Test edge travel time (empty link)
    if neighbors:
        tt = network.get_edge_travel_time(test_node, neighbors[0])
        print(f"Free-flow travel time {test_node} -> {neighbors[0]}: {tt:.2f}")

    # Test speed-density relationship
    print("\nSpeed-Density Relationship:")
    print(f"  {'Density':>8}  {'Speed':>8}  {'Travel Time':>12}")
    for d in range(0, 6):
        spd = speed_density(d, config.link_capacity, config.speed_limit)
        tt = compute_travel_time(d, config.link_capacity)
        print(f"  {d:>8}  {spd:>8.3f}  {tt:>12.3f}")

    # Test blocking a node
    network.block_node((3, 3))
    print(f"\nBlocked nodes: {network.get_blocked_nodes()}")
    print(f"Is (3,3) blocked? {network.is_node_blocked((3, 3))}")
    print(f"Is (2,2) blocked? {network.is_node_blocked((2, 2))}")

    # Test Euclidean distance (for communication radius)
    dist = network.euclidean_distance((0, 0), (0, 1))
    print(f"\nDistance between (0,0) and (0,1): {dist:.2f} blocks")
    dist2 = network.euclidean_distance((0, 0), (1, 1))
    print(f"Distance between (0,0) and (1,1): {dist2:.2f} blocks")

    # Visualize
    network.visualize(
        title="CAV Grid Network (6x6)",
        blocked_nodes=[(3, 3)],
        save_path="results/grid_network.png",
    )
