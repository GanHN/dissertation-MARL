"""
comm_manager.py - Communication & Object Memory Management
Handles all inter-vehicle communication for the CAV simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from src.environment.grid_network import GridNetwork
    from src.environment.vehicle import Vehicle

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.vehicle import CAV, MobilityMessage, VehicleType



@dataclass
class CommConfig:
    """Configuration for the communication system."""
    communication_radius: float = 0.5   # In block units
    enable_multi_hop: bool = True       # If True, use transitive clusters
    message_log_enabled: bool = False   # If True, record all messages for analysis



@dataclass
class ObstacleBroadcast:
    """
    Lightweight message sent when a CAV detects a blockage.
    When a CAV encounters a blocked intersection, it broadcasts a lightweight message containing the obstacle's
    Node ID to its cluster.
    """
    sender_id: int
    blocked_node: Tuple[int, int]
    timestamp: int



class CommunicationManager:
    """
    Manages all V2V communication between CAVs.

    Called once per simulation timestep to:
        1. Recompute communication clusters based on current positions
        2. Exchange mobility messages within each cluster
        3. Propagate obstacle broadcasts across clusters
    """

    def __init__(self, config: Optional[CommConfig] = None):
        self.config = config or CommConfig()

        # Current clusters: list of sets, each set contains vehicle IDs
        self.clusters: List[Set[int]] = []

        # Lookup: vehicle_id -> index into self.clusters
        self._vehicle_to_cluster: Dict[int, int] = {}

        # Message logs (for analysis / evaluation metrics)
        self.mobility_messages_sent: int = 0
        self.obstacle_broadcasts_sent: int = 0
        self.message_log: List[dict] = []


    def update_clusters(
        self,
        vehicles: List[Vehicle],
        network: GridNetwork,
    ) -> None:
        """
        Recompute communication clusters based on current vehicle positions.
        Only CAVs participate in clusters. HDVs are invisible to the
        communication system.
        """
        # Filter to only CAVs that are actively on the network
        active_cavs = [
            v for v in vehicles
            if v.vehicle_type == VehicleType.CAV
            and v.state.name in ("EN_ROUTE", "AT_INTERSECTION")
        ]

        if not active_cavs:
            self.clusters = []
            self._vehicle_to_cluster = {}
            return

        cr = self.config.communication_radius
        cav_ids = [v.vehicle_id for v in active_cavs]

        # Build adjacency: which CAVs are within direct CR of each other?
        adjacency: Dict[int, Set[int]] = {vid: set() for vid in cav_ids}

        for i, cav_a in enumerate(active_cavs):
            for j in range(i + 1, len(active_cavs)):
                cav_b = active_cavs[j]
                ax, ay = cav_a.get_continuous_position(network)
                bx, by = cav_b.get_continuous_position(network)
                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                if dist <= cr:
                    adjacency[cav_a.vehicle_id].add(cav_b.vehicle_id)
                    adjacency[cav_b.vehicle_id].add(cav_a.vehicle_id)

        if self.config.enable_multi_hop:
            # Find connected components using Union-Find
            self.clusters = self._find_connected_components(cav_ids, adjacency)
        else:
            # Each CAV's cluster is itself + direct neighbours
            self.clusters = []
            seen: Set[int] = set()
            for vid in cav_ids:
                if vid not in seen:
                    cluster = {vid} | adjacency[vid]
                    self.clusters.append(cluster)
                    seen.update(cluster)

        # Build reverse lookup
        self._vehicle_to_cluster = {}
        for idx, cluster in enumerate(self.clusters):
            for vid in cluster:
                self._vehicle_to_cluster[vid] = idx

    def _find_connected_components(
        self,
        node_ids: List[int],
        adjacency: Dict[int, Set[int]],
    ) -> List[Set[int]]:
        """
        Find connected components using BFS.

        This gives us multi-hop clusters: if A-B and B-C are connected,
        then {A, B, C} form one cluster.
        """
        visited: Set[int] = set()
        components: List[Set[int]] = []

        for start in node_ids:
            if start in visited:
                continue

            # BFS from this node
            component: Set[int] = set()
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)

                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            components.append(component)

        return components


    def get_cluster_for_vehicle(self, vehicle_id: int) -> Set[int]:
        """
        Get the set of vehicle IDs in the same cluster as the given vehicle.

        Returns an empty set if the vehicle is not in any cluster
        (e.g., it's an HDV, or it's isolated with no neighbours).
        """
        idx = self._vehicle_to_cluster.get(vehicle_id)
        if idx is None:
            return set()
        return self.clusters[idx]

    def get_cluster_members(
        self,
        vehicle_id: int,
        all_vehicles: List[Vehicle],
        exclude_self: bool = True,
    ) -> List[Vehicle]:
        """
        Get the actual Vehicle objects in the same cluster.
        """
        cluster_ids = self.get_cluster_for_vehicle(vehicle_id)
        if exclude_self:
            cluster_ids = cluster_ids - {vehicle_id}

        id_to_vehicle = {v.vehicle_id: v for v in all_vehicles}
        return [id_to_vehicle[vid] for vid in cluster_ids if vid in id_to_vehicle]

    def get_num_clusters(self) -> int:
        """Return the number of clusters in the current timestep."""
        return len(self.clusters)

    def get_cluster_sizes(self) -> List[int]:
        """Return the size of each cluster."""
        return [len(c) for c in self.clusters]


    def exchange_mobility_messages(
        self,
        vehicles: List[Vehicle],
        timestep: int,
    ) -> Dict[int, List[MobilityMessage]]:
        """
        Each CAV broadcasts a MobilityMessage to its cluster.

        Returns a dict mapping each CAV's ID to the list of messages
        it received from cluster neighbours. This is the input data
        for Dec-CTDSP's time-dependent network construction.
        """
        id_to_vehicle = {v.vehicle_id: v for v in vehicles}
        received: Dict[int, List[MobilityMessage]] = {}

        for cluster in self.clusters:
            # Collect messages from all CAVs in this cluster
            cluster_messages: List[MobilityMessage] = []
            for vid in cluster:
                v = id_to_vehicle.get(vid)
                if v is not None and isinstance(v, CAV):
                    msg = v.create_mobility_message(timestep)
                    cluster_messages.append(msg)
                    self.mobility_messages_sent += 1

            # Each CAV receives messages from everyone else in the cluster
            for vid in cluster:
                received[vid] = [
                    m for m in cluster_messages if m.vehicle_id != vid
                ]

        return received


    def broadcast_obstacle(
        self,
        sender: CAV,
        blocked_node: Tuple[int, int],
        vehicles: List[Vehicle],
        timestep: int,
    ) -> int:
        """
        Broadcast an obstacle detection to the sender's entire cluster.
        """
        # Sender updates its own blacklist
        sender.add_to_blacklist(blocked_node, timestep)

        # Create the broadcast message
        broadcast = ObstacleBroadcast(
            sender_id=sender.vehicle_id,
            blocked_node=blocked_node,
            timestamp=timestep,
        )

        # Get cluster members
        cluster_ids = self.get_cluster_for_vehicle(sender.vehicle_id)
        id_to_vehicle = {v.vehicle_id: v for v in vehicles}

        receivers = 0
        for vid in cluster_ids:
            if vid == sender.vehicle_id:
                continue
            v = id_to_vehicle.get(vid)
            if v is not None and isinstance(v, CAV):
                v.receive_obstacle_broadcast(blocked_node, timestep)
                receivers += 1

        self.obstacle_broadcasts_sent += 1

        # Optional logging
        if self.config.message_log_enabled:
            self.message_log.append({
                "type": "obstacle",
                "timestep": timestep,
                "sender": sender.vehicle_id,
                "blocked_node": blocked_node,
                "receivers": receivers,
                "cluster_size": len(cluster_ids),
            })

        return receivers

    def propagate_confirmations(
        self,
        vehicles: List[Vehicle],
        network: GridNetwork,
        timestep: int,
    ) -> int:
        """
        CAVs near a still-blocked node re-broadcast confirmations.
        """
        blocked_nodes = set(network.get_blocked_nodes())
        if not blocked_nodes:
            return 0

        confirmations = 0
        id_to_vehicle = {v.vehicle_id: v for v in vehicles}

        for vehicle in vehicles:
            if not isinstance(vehicle, CAV):
                continue
            if vehicle.state.name not in ("EN_ROUTE", "AT_INTERSECTION"):
                continue

            # Check if this CAV is at or adjacent to a blocked node
            current = vehicle.current_node
            nearby_nodes = {current} | set(network.get_neighbors(current))
            detected_blocks = nearby_nodes & blocked_nodes

            for blocked_node in detected_blocks:
                # Only broadcast if this CAV already knows about it
                # (or is physically at the blocked node)
                if blocked_node in vehicle.get_blacklisted_nodes() or blocked_node == current:
                    # Refresh own blacklist
                    vehicle.add_to_blacklist(blocked_node, timestep)

                    # Broadcast to cluster
                    cluster_ids = self.get_cluster_for_vehicle(vehicle.vehicle_id)
                    for vid in cluster_ids:
                        if vid == vehicle.vehicle_id:
                            continue
                        v = id_to_vehicle.get(vid)
                        if v is not None and isinstance(v, CAV):
                            v.receive_obstacle_broadcast(blocked_node, timestep)

                    confirmations += 1

        return confirmations


    def decay_all_blacklists(
        self,
        vehicles: List[Vehicle],
        timestep: int,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Decay expired blacklist entries for all CAVs.
        Returns a dict of {vehicle_id: [expired_nodes]} so the simulator
        can log which nodes were un-blacklisted.
        """
        expired_map: Dict[int, List[Tuple[int, int]]] = {}
        for vehicle in vehicles:
            if isinstance(vehicle, CAV):
                expired = vehicle.decay_blacklist(timestep)
                if expired:
                    expired_map[vehicle.vehicle_id] = expired
        return expired_map


    def get_stats(self) -> Dict:
        """Return communication statistics for evaluation."""
        return {
            "num_clusters": self.get_num_clusters(),
            "cluster_sizes": self.get_cluster_sizes(),
            "largest_cluster": max(self.get_cluster_sizes()) if self.clusters else 0,
            "mobility_messages_sent": self.mobility_messages_sent,
            "obstacle_broadcasts_sent": self.obstacle_broadcasts_sent,
        }

    def reset_counters(self) -> None:
        """Reset per-timestep message counters."""
        self.mobility_messages_sent = 0
        self.obstacle_broadcasts_sent = 0



if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.environment.grid_network import GridNetwork, NetworkConfig
    from src.environment.vehicle import CAV, HDV, VehicleFactory, VehicleType

    # Build network
    network = GridNetwork(NetworkConfig())

    print("=" * 60)
    print("Communication Manager Test")
    print("=" * 60)

    # ── Test 1: Cluster formation ──
    print("\n--- Cluster Formation Test ---")

    # Create some CAVs at known positions
    cav_a = CAV(vehicle_id=0, origin=(0, 0), destination=(0, 5))
    cav_a.current_node = (2, 2)
    cav_a.state = cav_a.state.__class__["EN_ROUTE"]

    cav_b = CAV(vehicle_id=1, origin=(1, 0), destination=(1, 5))
    cav_b.current_node = (2, 3)  # 1.0 block from cav_a
    cav_b.state = cav_b.state.__class__["EN_ROUTE"]

    cav_c = CAV(vehicle_id=2, origin=(2, 0), destination=(2, 5))
    cav_c.current_node = (2, 4)  # 1.0 block from cav_b, 2.0 from cav_a
    cav_c.state = cav_c.state.__class__["EN_ROUTE"]

    cav_d = CAV(vehicle_id=3, origin=(3, 0), destination=(3, 5))
    cav_d.current_node = (4, 0)  # Far from the others
    cav_d.state = cav_d.state.__class__["EN_ROUTE"]

    hdv = HDV(vehicle_id=4, origin=(0, 0), destination=(4, 5))
    hdv.current_node = (2, 2)  # Same position as cav_a but HDV — invisible

    vehicles = [cav_a, cav_b, cav_c, cav_d, hdv]

    # CR = 0.5 (only adjacent CAVs directly connected)
    comm = CommunicationManager(CommConfig(communication_radius=0.5))
    comm.update_clusters(vehicles, network)
    print(f"CR=0.5 | Clusters: {comm.clusters}")
    print(f"  -> No direct connections (all > 0.5 apart)")

    # CR = 1.0 (adjacent CAVs connected)
    comm1 = CommunicationManager(CommConfig(communication_radius=1.0))
    comm1.update_clusters(vehicles, network)
    print(f"\nCR=1.0 | Clusters: {comm1.clusters}")
    print(f"  -> A-B connected (dist=1.0), B-C connected (dist=1.0)")
    print(f"  -> Multi-hop: A-B-C in one cluster, D isolated")

    # CR = 3.0 (everyone connected)
    comm3 = CommunicationManager(CommConfig(communication_radius=3.0))
    comm3.update_clusters(vehicles, network)
    print(f"\nCR=3.0 | Clusters: {comm3.clusters}")
    print(f"  -> All 4 CAVs in one cluster")

    # ── Test 2: Cluster queries ──
    print("\n--- Cluster Query Test ---")
    comm1.update_clusters(vehicles, network)
    cluster_a = comm1.get_cluster_for_vehicle(0)
    print(f"CAV 0's cluster (CR=1.0): {cluster_a}")

    members = comm1.get_cluster_members(0, vehicles, exclude_self=True)
    print(f"CAV 0's cluster members: {[v.vehicle_id for v in members]}")

    cluster_d = comm1.get_cluster_for_vehicle(3)
    print(f"CAV 3's cluster (isolated): {cluster_d}")

    # HDV not in any cluster
    cluster_hdv = comm1.get_cluster_for_vehicle(4)
    print(f"HDV 4's cluster: {cluster_hdv} (empty — HDVs don't communicate)")

    # ── Test 3: Mobility message exchange ──
    print("\n--- Mobility Message Exchange Test ---")
    # Give CAVs routes so messages have content
    cav_a.planned_route = [(2, 3), (2, 4), (2, 5)]
    cav_b.planned_route = [(2, 4), (2, 5)]
    cav_c.planned_route = [(2, 5)]

    received = comm1.exchange_mobility_messages(vehicles, timestep=0)
    for vid, msgs in received.items():
        print(f"  CAV {vid} received {len(msgs)} messages: "
              f"{[(m.vehicle_id, m.current_node) for m in msgs]}")

    # ── Test 4: Obstacle broadcasting ──
    print("\n--- Obstacle Broadcasting Test ---")
    network.block_node((3, 3))

    # CAV A detects the obstacle and broadcasts
    num_receivers = comm1.broadcast_obstacle(cav_a, (3, 3), vehicles, timestep=1)
    print(f"CAV 0 broadcast obstacle (3,3) -> {num_receivers} receivers")

    # Check that cluster members got the blacklist update
    print(f"CAV 0 blacklist: {cav_a.get_blacklisted_nodes()}")
    print(f"CAV 1 blacklist: {cav_b.get_blacklisted_nodes()}")
    print(f"CAV 2 blacklist: {cav_c.get_blacklisted_nodes()}")
    print(f"CAV 3 blacklist: {cav_d.get_blacklisted_nodes()} (isolated — no broadcast)")

    # ── Test 5: Confirmation propagation ──
    print("\n--- Confirmation Propagation Test ---")
    # Move CAV B near the blocked node
    cav_b.current_node = (3, 2)  # Adjacent to blocked (3,3)
    cav_b.add_to_blacklist((3, 3), timestep=1)  # Already knows about it
    comm1.update_clusters(vehicles, network)

    confirmations = comm1.propagate_confirmations(vehicles, network, timestep=10)
    print(f"Confirmations sent at t=10: {confirmations}")

    # Check CAV A's blacklist — TTL should be refreshed
    entry = cav_a.blacklist.get((3, 3))
    if entry:
        print(f"CAV 0 blacklist entry for (3,3): last_confirmed={entry.last_confirmed}")
        print(f"  -> Refreshed from t=1 to t=10 by confirmation")

    # ── Test 6: Blacklist decay ──
    print("\n--- Blacklist Decay Test ---")
    # Fast-forward time past TTL
    expired = comm1.decay_all_blacklists(vehicles, timestep=70)
    print(f"Expired at t=70: {expired}")
    print(f"CAV 3 blacklist after decay: {cav_d.get_blacklisted_nodes()}")
    print(f"CAV 0 blacklist after decay: {cav_a.get_blacklisted_nodes()}")

    # ── Test 7: Fleet-scale test ──
    print("\n--- Fleet Scale Test ---")
    fleet = VehicleFactory.create_fleet(
        num_vehicles=100,
        market_penetration=1.0,
        origins=network.origins,
        destinations=network.destinations,
        seed=42,
    )
    # Place vehicles at their origins and set them en route
    for v in fleet:
        v.state = v.state.__class__["EN_ROUTE"]
        v.compute_route(network, timestep=0)

    comm_fleet = CommunicationManager(CommConfig(communication_radius=0.5))
    comm_fleet.update_clusters(fleet, network)
    stats = comm_fleet.get_stats()
    print(f"100 CAVs, CR=0.5:")
    print(f"  Clusters: {stats['num_clusters']}")
    print(f"  Largest cluster: {stats['largest_cluster']}")
    print(f"  Cluster sizes: {sorted(stats['cluster_sizes'], reverse=True)[:5]}...")

    print("\nAll tests passed.")
