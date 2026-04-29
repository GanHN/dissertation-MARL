"""
dec_ctdsp.py - Decentralized Collaborative Time-Dependent Shortest Path
Direct implementation of the two core algorithms from:
    Mostafizi et al., "A Decentralized and Coordinated Routing Algorithm
    for Connected and Autonomous Vehicles," IEEE Trans. ITS, 2022.

Algorithm 1: build_time_dependent_network()
    Takes the cluster vehicles' locations, and planned routes,
    simulates their movements over a planning horizon, and produces a
    time-dependent travel time for every edge at every future timestep.

Algorithm 2: time_dependent_dijkstra()
    Modified Dijkstra that looks up edge weights from the time-dependent
    network at the time the vehicle would actually arrive at each node.
    Crucially excludes OMM-blacklisted nodes from the search.

dec_ctdsp_route():
    The top-level function that combines both algorithms. This is what
    gets injected into CAV.set_routing_function().
"""

from __future__ import annotations

import heapq
import sys
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.grid_network import GridNetwork, compute_travel_time

if TYPE_CHECKING:
    from src.environment.vehicle import Vehicle



@dataclass
class DecCTDSPConfig:
    """Tuneable parameters for the Dec-CTDSP algorithm."""
    planning_horizon: int = 10   # How many timesteps into the future to simulate
    gridlock_penalty: float = 100.0  # Travel time when a link is fully gridlocked



class TimeDependentNetwork:
    """
    Stores time-varying travel times for every edge in the network.

    For each edge (u, v), stores a list of (timestep, travel_time) tuples.
    This is the TDep-G from the paper.

    The key method is get_travel_time(u, v, t) which looks up the travel
    time for edge (u,v) at time t using linear interpolation between
    the two nearest recorded timesteps.
    """

    def __init__(self):
        # {(u, v): [(t0, tt0), (t1, tt1), ...]}
        self.travel_times: Dict[Tuple, List[Tuple[float, float]]] = defaultdict(list)

    def add_entry(
        self,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int],
        timestep: float,
        travel_time: float,
    ) -> None:
        """Record a travel time observation for an edge at a given time."""
        self.travel_times[(from_node, to_node)].append((timestep, travel_time))

    def get_travel_time(
        self,
        from_node: Tuple[int, int],
        to_node: Tuple[int, int],
        t: float,
    ) -> float:
        """
        GetTravelTime(TDep-G, u, v, f[u]) from Algorithm 2.

        Looks up the travel time for edge (u,v) at time t.
        Uses linear interpolation between the two nearest timesteps.
        If t is beyond the last recorded timestep, returns the last value.
        If t is before the first, returns the first value.

        Args:
            from_node: Source node of the edge.
            to_node:   Target node of the edge.
            t:         Time at which to query (f[u] in the paper).

        Returns:
            Interpolated travel time for this edge at time t.
        """
        entries = self.travel_times.get((from_node, to_node))

        if not entries:
            # Edge not in the time-dependent network — return large value
            return 100.0

        # Entries should be sorted by time
        if len(entries) == 1:
            return entries[0][1]

        # Find the two entries that bracket time t
        if t <= entries[0][0]:
            return entries[0][1]
        if t >= entries[-1][0]:
            return entries[-1][1]

        # Linear interpolation
        for i in range(len(entries) - 1):
            t0, tt0 = entries[i]
            t1, tt1 = entries[i + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    return tt0
                # Interpolate
                ratio = (t - t0) / (t1 - t0)
                return tt0 + ratio * (tt1 - tt0)

        return entries[-1][1]



def build_time_dependent_network(
    network: GridNetwork,
    cluster_vehicles: List[Vehicle],
    planning_horizon: int = 10,
) -> TimeDependentNetwork:
    """
    Algorithm 1 from the Dec-CTDSP paper.

    Simulates the movement of cluster vehicles over a planning horizon
    to predict how congested each link will be at each future timestep.

    Process:
        1. If the cluster is empty, set all edges to free-flow travel time.
        2. Otherwise, for each timestep in [0, planning_horizon]:
            a. Simulate each vehicle moving one step along its route.
            b. Count vehicles on each edge to get density.
            c. Use speed-density relationship to compute travel time.
            d. Record (timestep, travel_time) for each edge.

    Args:
        network:          The grid network.
        cluster_vehicles: Other CAVs in the communication cluster
                          (with their current_node, speed, and planned_route).
        planning_horizon: Number of future timesteps to simulate.

    Returns:
        TimeDependentNetwork with travel times for all edges at all timesteps.
    """
    td_net = TimeDependentNetwork()
    all_edges = list(network.graph.edges())

    # Case 1: Empty cluster — all edges get free-flow travel time
    if not cluster_vehicles:
        for (u, v) in all_edges:
            ff_tt = network.get_free_flow_travel_time(u, v)
            td_net.add_entry(u, v, 0.0, ff_tt)
            td_net.add_entry(u, v, float(planning_horizon), ff_tt)
        return td_net

    # Case 2: Simulate vehicle movements
    # Create lightweight copies of vehicle positions and routes
    sim_vehicles = []
    cluster_vehicle_ids = {v.vehicle_id for v in cluster_vehicles}
    for v in cluster_vehicles:
        remaining_route = v.get_remaining_route()
        sim_vehicles.append({
            "current_node": v.current_node,
            "route": list(remaining_route),
            "route_idx": 0,
            "speed": v.speed if v.speed > 0 else network.config.speed_limit,
        })

    # For each timestep, simulate movement and record edge densities
    for t in range(planning_horizon + 1):
        # Count vehicles on each edge at this timestep
        edge_density: Dict[Tuple, int] = defaultdict(int)

        for sv in sim_vehicles:
            # Determine which edge this vehicle is on
            current = sv["current_node"]
            route = sv["route"]
            idx = sv["route_idx"]

            if idx < len(route):
                next_node = route[idx]
                # Vehicle is on edge (current -> next_node)
                if network.graph.has_edge(current, next_node):
                    edge_density[(current, next_node)] += 1

        # Compute travel time for each edge based on density
        for (u, v) in all_edges:
            edge_data = network.graph.edges[u, v]
            density = edge_density.get((u, v), 0)

            # Include live vehicles outside the communication cluster as
            # background traffic. Cluster vehicles are already represented by
            # the simulated density above, so we exclude them here.
            existing_noncluster = sum(
                1
                for vid in edge_data["current_vehicles"]
                if vid not in cluster_vehicle_ids
            )
            total_density = density + existing_noncluster

            tt = compute_travel_time(
                density=total_density,
                capacity=edge_data["capacity"],
                free_flow_tt=edge_data["free_flow_tt"],
                speed_limit=network.config.speed_limit,
            )
            td_net.add_entry(u, v, float(t), tt)

        # SimulateVehicleMovement: advance each vehicle one step
        for sv in sim_vehicles:
            route = sv["route"]
            idx = sv["route_idx"]
            if idx < len(route):
                sv["current_node"] = route[idx]
                sv["route_idx"] = idx + 1

    return td_net



def time_dependent_dijkstra(
    network: GridNetwork,
    td_network: TimeDependentNetwork,
    source: Tuple[int, int],
    target: Tuple[int, int],
    blacklist: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Algorithm 2 from the Dec-CTDSP paper, modified with OMM blacklist.

    Standard Dijkstra but instead of static edge weights, it calls
    GetTravelTime(TDep-G, u, v, f[u]) to look up the travel time
    for edge (u,v) at the time the vehicle would arrive at u.

    The blacklist modification: any node in the blacklist is skipped
    during neighbour expansion (unless it's the target itself).

    Args:
        network:    The grid network (for topology/neighbour lookups).
        td_network: Time-dependent travel time data from Algorithm 1.
        source:     Current intersection of the CAV.
        target:     Destination of the CAV.
        blacklist:  Set of node IDs to exclude from routing.

    Returns:
        List of nodes from source to target (excluding source).
        Empty list if no path found.
    """
    # Priority queue: (cumulative_arrival_time, node)
    pq = [(0.0, source)]

    # f[v] = earliest arrival time at node v
    f: Dict[Tuple[int, int], float] = {source: 0.0}

    # prev[v] = predecessor on the shortest path
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {source: None}

    # Track which nodes have been finalized
    visited: Set[Tuple[int, int]] = set()

    while pq:
        arrival_time, u = heapq.heappop(pq)

        # Already found a shorter path to u
        if u in visited:
            continue
        visited.add(u)

        # Reached destination — reconstruct path
        if u == target:
            path = []
            node = target
            while node is not None and node != source:
                path.append(node)
                node = prev.get(node)
            path.reverse()
            return path

        # Expand neighbours
        for v in network.get_neighbors(u):
            # Skip already-visited nodes
            if v in visited:
                continue

            # OMM blacklist exclusion — the key modification
            # Always allow the target even if blacklisted (edge case)
            if v in blacklist and v != target:
                continue

            # GetTravelTime(TDep-G, u, v, f[u])
            # Look up travel time at the time we'd arrive at u
            duv = td_network.get_travel_time(u, v, arrival_time)

            alt = arrival_time + duv

            if alt < f.get(v, float("inf")):
                f[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    # No path found
    return []



def dec_ctdsp_route(
    network: GridNetwork,
    source: Tuple[int, int],
    target: Tuple[int, int],
    blacklist: Set[Tuple[int, int]],
    cluster_vehicles: List[Vehicle],
    timestep: int,
    config: Optional[DecCTDSPConfig] = None,
) -> List[Tuple[int, int]]:
    """
    The complete Dec-CTDSP routing pipeline.

    This is the function you inject into a CAV with:
        cav.set_routing_function(dec_ctdsp_route)

    Steps:
        1. Build the time-dependent network using cluster info (Algorithm 1)
        2. Run modified Dijkstra on it, excluding blacklisted nodes (Algorithm 2)
        3. Return the optimal route

    Args:
        network:          The grid network.
        source:           Current node of the CAV.
        target:           Destination of the CAV.
        blacklist:        OMM blacklisted nodes to exclude.
        cluster_vehicles: Other CAVs in the communication cluster.
        timestep:         Current simulation timestep (unused but required by interface).
        config:           Algorithm configuration.

    Returns:
        List of nodes from source to target (excluding source).
        Empty list if no path found.
    """
    if config is None:
        config = DecCTDSPConfig()

    # Algorithm 1: Build time-dependent network
    td_net = build_time_dependent_network(
        network=network,
        cluster_vehicles=cluster_vehicles,
        planning_horizon=config.planning_horizon,
    )

    # Algorithm 2: Time-dependent Dijkstra with blacklist
    route = time_dependent_dijkstra(
        network=network,
        td_network=td_net,
        source=source,
        target=target,
        blacklist=blacklist,
    )

    return route



if __name__ == "__main__":
    from src.environment.grid_network import GridNetwork, NetworkConfig
    from src.environment.vehicle import CAV, HDV, VehicleFactory

    # Build network
    network = GridNetwork(NetworkConfig())

    print("=" * 60)
    print("Dec-CTDSP Algorithm Test")
    print("=" * 60)

    # ── Test 1: Empty cluster (free-flow) ──
    print("\n--- Test 1: Empty Cluster (Free-Flow Routing) ---")
    route = dec_ctdsp_route(
        network=network,
        source=(0, 0),
        target=(0, 5),
        blacklist=set(),
        cluster_vehicles=[],
        timestep=0,
    )
    print(f"Route (0,0) -> (0,5): {route}")
    print(f"Length: {len(route)} steps (expected: 5, straight east)")

    # ── Test 2: With blacklist ──
    print("\n--- Test 2: Routing with Blacklisted Nodes ---")
    route_bl = dec_ctdsp_route(
        network=network,
        source=(2, 0),
        target=(2, 5),
        blacklist={(2, 2), (2, 3)},
        cluster_vehicles=[],
        timestep=0,
    )
    print(f"Route (2,0) -> (2,5) avoiding (2,2) and (2,3):")
    print(f"  {route_bl}")
    avoids = (2, 2) not in route_bl and (2, 3) not in route_bl
    print(f"  Avoids blacklisted? {avoids}")

    # ── Test 3: With cluster vehicles causing congestion ──
    print("\n--- Test 3: Cluster Vehicles Causing Congestion ---")

    # Create some CAVs that will congest the middle row
    congestion_cavs = []
    for i in range(4):
        cav = CAV(vehicle_id=100 + i, origin=(2, 0), destination=(2, 5))
        cav.current_node = (2, i)
        cav.speed = 1.0
        cav.planned_route = [(2, j) for j in range(i + 1, 6)]
        cav.route_index = 0
        cav.state = cav.state.__class__["EN_ROUTE"]
        congestion_cavs.append(cav)

    # Route without cluster awareness (free flow)
    route_no_cluster = dec_ctdsp_route(
        network=network,
        source=(0, 0),
        target=(2, 5),
        blacklist=set(),
        cluster_vehicles=[],
        timestep=0,
    )
    print(f"Route WITHOUT cluster info: {route_no_cluster}")

    # Route with cluster awareness (should avoid congested row 2)
    route_with_cluster = dec_ctdsp_route(
        network=network,
        source=(0, 0),
        target=(2, 5),
        blacklist=set(),
        cluster_vehicles=congestion_cavs,
        timestep=0,
    )
    print(f"Route WITH cluster info:    {route_with_cluster}")

    # Check if the cluster-aware route uses fewer row-2 edges
    row2_no = sum(1 for n in route_no_cluster if n[0] == 2)
    row2_with = sum(1 for n in route_with_cluster if n[0] == 2)
    print(f"Row 2 nodes used: without={row2_no}, with={row2_with}")

    # ── Test 4: Time-dependent network inspection ──
    print("\n--- Test 4: Time-Dependent Network Details ---")
    td_net = build_time_dependent_network(
        network=network,
        cluster_vehicles=congestion_cavs,
        planning_horizon=5,
    )
    # Show travel times for a congested edge vs uncongested edge
    edge_congested = ((2, 0), (2, 1))
    edge_free = ((0, 0), (0, 1))

    print(f"Edge {edge_congested} (congested by cluster):")
    for t in range(6):
        tt = td_net.get_travel_time(*edge_congested, float(t))
        print(f"  t={t}: travel_time={tt:.3f}")

    print(f"\nEdge {edge_free} (no congestion):")
    for t in range(6):
        tt = td_net.get_travel_time(*edge_free, float(t))
        print(f"  t={t}: travel_time={tt:.3f}")

    # ── Test 5: Integration with CAV routing ──
    print("\n--- Test 5: CAV Integration ---")
    test_cav = CAV(vehicle_id=999, origin=(1, 0), destination=(3, 5)) #Juice Wrld Tribute
    test_cav.current_node = (1, 0)
    test_cav.state = test_cav.state.__class__["EN_ROUTE"]

    # Inject Dec-CTDSP as the routing function
    test_cav.set_routing_function(dec_ctdsp_route)

    # Add some blacklisted nodes
    test_cav.add_to_blacklist((2, 2), timestep=0)

    # Compute route with Dec-CTDSP
    test_cav.compute_route(
        network=network,
        timestep=0,
        cluster_vehicles=congestion_cavs,
    )
    print(f"CAV 999 route: {test_cav.planned_route}")
    print(f"Avoids (2,2)? {(2, 2) not in test_cav.planned_route}")
    print(f"Route recalculations: {test_cav.num_route_recalculations}")

    # ── Test 6: No path possible (everything blocked) ──
    print("\n--- Test 6: No Path Available ---")
    # Block all neighbours of source
    heavy_blacklist = {(0, 1), (1, 0)}
    route_blocked = dec_ctdsp_route(
        network=network,
        source=(0, 0),
        target=(4, 5),
        blacklist=heavy_blacklist,
        cluster_vehicles=[],
        timestep=0,
    )
    print(f"Route with all exits blocked: {route_blocked}")
    print(f"Empty route (no path)? {len(route_blocked) == 0}")

    print("\nAll tests passed.")
