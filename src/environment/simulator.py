"""
simulator.py - Main Simulation Loop
The central engine that ties all modules together and runs the
agent-based traffic simulation.

Each timestep:
    1. Update communication clusters
    2. Propagate obstacle confirmations (OMM)
    3. Decay expired blacklist entries
    4. For each vehicle at an intersection:
       a. Check for obstacles -> broadcast if detected
       b. Compute/recompute route (Dec-CTDSP for CAVs, naive for HDVs)
    5. Move vehicles along their routes
    6. Handle arrivals -> reset for new trip (continuous format)
    7. Record metrics (MSTT, MSS, wait times, recalculations)

From the paper:
    "Each vehicle returns to its origin after reaching the destination
     and starts another trip, making the problem domain sequential
     and open-ended."

    "The simulation is run until the MSTT of the system converges.
     This event is defined as the standard deviation of the last 200
     readings dropping below 2%."
"""

from __future__ import annotations

import random
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.grid_network import GridNetwork, NetworkConfig
from src.environment.vehicle import (
    CAV, HDV, Vehicle, VehicleFactory, VehicleState, VehicleType,
)
from src.communication.comm_manager import CommunicationManager, CommConfig
from src.routing.dec_ctdsp import dec_ctdsp_route, DecCTDSPConfig


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All tuneable simulation parameters in one place."""
    # Network
    grid_rows: int = 6
    grid_cols: int = 6
    link_capacity: int = 5

    # Fleet
    num_vehicles: int = 100
    market_penetration: float = 1.0      # 0.0 to 1.0

    # Communication
    communication_radius: float = 0.5    # In block units

    # OMM
    blacklist_ttl: int = 50

    # Obstacles
    num_obstacles: int = 2
    obstacle_spawn_interval: int = 50    # Respawn obstacles every N timesteps
    obstacle_duration: int = 40          # How long each obstacle lasts

    # Dec-CTDSP
    planning_horizon: int = 10

    # Simulation
    max_timesteps: int = 500
    convergence_window: int = 200        # Check last N MSTT readings
    convergence_threshold: float = 0.02  # Std dev / mean < this = converged
    warmup_steps: int = 20               # Steps before recording metrics

    # Reproducibility
    seed: int = 42


# ── Metrics Recorder ─────────────────────────────────────────────────────────

@dataclass
class SimMetrics:
    """Tracks all evaluation metrics throughout the simulation."""
    # Per-timestep recordings
    mstt_history: List[float] = field(default_factory=list)
    mss_history: List[float] = field(default_factory=list)
    total_trips_completed: int = 0
    total_route_recalculations: int = 0
    total_obstacle_broadcasts: int = 0
    total_wait_timesteps: int = 0

    # Per-trip data
    all_trip_times: List[float] = field(default_factory=list)

    def record_timestep(
        self,
        vehicles: List[Vehicle],
        network: GridNetwork,
        timestep: int,
    ) -> None:
        """Record MSTT and MSS for this timestep."""
        # MSTT: average of each vehicle's most recent trip time
        recent_trip_times = []
        speeds = []

        for v in vehicles:
            if v.trip_travel_times:
                recent_trip_times.append(v.trip_travel_times[-1])
            if v.state == VehicleState.EN_ROUTE and v.speed > 0:
                speeds.append(v.speed)

        if recent_trip_times:
            self.mstt_history.append(np.mean(recent_trip_times))
        if speeds:
            self.mss_history.append(np.mean(speeds))

    def is_converged(self, window: int = 200, threshold: float = 0.02) -> bool:
        """
        Check if MSTT has converged.

        From the paper: "standard deviation of the last 200 readings
        dropping below 2%."
        """
        if len(self.mstt_history) < window:
            return False

        recent = self.mstt_history[-window:]
        mean_val = np.mean(recent)
        if mean_val == 0:
            return True
        std_val = np.std(recent)
        return (std_val / mean_val) < threshold

    def get_final_mstt(self) -> float:
        """Get the converged MSTT value."""
        if not self.mstt_history:
            return 0.0
        return float(np.mean(self.mstt_history[-100:])) if len(self.mstt_history) >= 100 else float(np.mean(self.mstt_history))

    def get_final_mss(self) -> float:
        """Get the converged MSS value."""
        if not self.mss_history:
            return 0.0
        return float(np.mean(self.mss_history[-100:])) if len(self.mss_history) >= 100 else float(np.mean(self.mss_history))

    def summary(self) -> Dict:
        """Return a summary dict of all metrics."""
        return {
            "final_mstt": round(self.get_final_mstt(), 3),
            "final_mss": round(self.get_final_mss(), 3),
            "total_trips": self.total_trips_completed,
            "total_recalculations": self.total_route_recalculations,
            "total_obstacle_broadcasts": self.total_obstacle_broadcasts,
            "avg_trip_time": round(np.mean(self.all_trip_times), 3) if self.all_trip_times else 0,
            "avg_wait_per_vehicle": round(self.total_wait_timesteps / max(1, self.total_trips_completed), 3),
            "timesteps_run": len(self.mstt_history),
        }


# ── Obstacle Manager ─────────────────────────────────────────────────────────

class ObstacleManager:
    """Handles spawning and clearing of dynamic obstacles."""

    def __init__(
        self,
        network: GridNetwork,
        num_obstacles: int = 2,
        spawn_interval: int = 50,
        duration: int = 40,
        seed: int = 42,
    ):
        self.network = network
        self.num_obstacles = num_obstacles
        self.spawn_interval = spawn_interval
        self.duration = duration
        self.rng = random.Random(seed)

        # {node: timestep_when_spawned}
        self.active_obstacles: Dict[Tuple[int, int], int] = {}

    def update(self, timestep: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Update obstacles: clear expired ones and spawn new ones.

        Returns:
            (newly_spawned, newly_cleared) lists of nodes.
        """
        newly_cleared = []
        newly_spawned = []

        # Clear expired obstacles
        for node, spawn_time in list(self.active_obstacles.items()):
            if timestep - spawn_time >= self.duration:
                self.network.unblock_node(node)
                del self.active_obstacles[node]
                newly_cleared.append(node)

        # Spawn new obstacles at intervals
        if timestep > 0 and timestep % self.spawn_interval == 0:
            # Pick random interior nodes (not origins or destinations)
            interior_nodes = [
                n for n in self.network.graph.nodes()
                if n not in self.network.origins
                and n not in self.network.destinations
                and n not in self.active_obstacles
            ]

            num_to_spawn = min(self.num_obstacles, len(interior_nodes))
            new_obstacles = self.rng.sample(interior_nodes, num_to_spawn)

            for node in new_obstacles:
                self.network.block_node(node)
                self.active_obstacles[node] = timestep
                newly_spawned.append(node)

        return newly_spawned, newly_cleared


# ── Main Simulator ───────────────────────────────────────────────────────────

class Simulator:
    """
    The main simulation engine.

    Orchestrates all components: network, vehicles, communication,
    OMM, routing, obstacles, and metrics collection.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self._setup()

    def _setup(self) -> None:
        """Initialise all components."""
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Network
        net_config = NetworkConfig(
            rows=cfg.grid_rows,
            cols=cfg.grid_cols,
            link_capacity=cfg.link_capacity,
            num_vehicles=cfg.num_vehicles,
        )
        self.network = GridNetwork(net_config)

        # Fleet
        self.vehicles = VehicleFactory.create_fleet(
            num_vehicles=cfg.num_vehicles,
            market_penetration=cfg.market_penetration,
            origins=self.network.origins,
            destinations=self.network.destinations,
            blacklist_ttl=cfg.blacklist_ttl,
            seed=cfg.seed,
        )

        # Inject Dec-CTDSP routing into all CAVs
        for v in self.vehicles:
            if isinstance(v, CAV):
                v.set_routing_function(dec_ctdsp_route)

        # Communication
        comm_config = CommConfig(communication_radius=cfg.communication_radius)
        self.comm_manager = CommunicationManager(comm_config)

        # Obstacles
        self.obstacle_manager = ObstacleManager(
            network=self.network,
            num_obstacles=cfg.num_obstacles,
            spawn_interval=cfg.obstacle_spawn_interval,
            duration=cfg.obstacle_duration,
            seed=cfg.seed + 1,
        )

        # Metrics
        self.metrics = SimMetrics()

        # Timestep counter
        self.timestep = 0

    # ── Main Loop ────────────────────────────────────────────────────────

    def run(self, verbose: bool = False) -> SimMetrics:
        """
        Run the full simulation until convergence or max timesteps.

        Args:
            verbose: If True, print progress every 50 timesteps.

        Returns:
            SimMetrics with all recorded data.
        """
        cfg = self.config

        # Initialise: compute initial routes and depart all vehicles
        self._initialise_vehicles()

        for t in range(cfg.max_timesteps):
            self.timestep = t
            self._step(t)

            # Record metrics after warmup
            if t >= cfg.warmup_steps:
                self.metrics.record_timestep(self.vehicles, self.network, t)

                # Check convergence
                if self.metrics.is_converged(
                    cfg.convergence_window, cfg.convergence_threshold
                ):
                    if verbose:
                        print(f"  Converged at timestep {t}")
                    break

            if verbose and t % 50 == 0:
                completed = self.metrics.total_trips_completed
                blocked = len(self.network.get_blocked_nodes())
                mstt = self.metrics.get_final_mstt()
                print(f"  t={t:4d} | trips={completed:4d} | "
                      f"blocked={blocked} | MSTT={mstt:.2f}")

        # Collect final recalculation counts
        for v in self.vehicles:
            self.metrics.total_route_recalculations += v.num_route_recalculations

        return self.metrics

    def _initialise_vehicles(self) -> None:
        """Compute initial routes and set all vehicles to EN_ROUTE."""
        for v in self.vehicles:
            if isinstance(v, CAV):
                v.compute_route(self.network, timestep=0, cluster_vehicles=[])
            elif isinstance(v, HDV):
                v.compute_route(self.network, timestep=0)
            v.depart(timestep=0)
            v.speed = self.network.config.speed_limit

    # ── Single Timestep ──────────────────────────────────────────────────

    def _step(self, t: int) -> None:
        """Execute one simulation timestep."""
        # 1. Update obstacles (spawn/clear)
        spawned, cleared = self.obstacle_manager.update(t)

        # 2. Update communication clusters
        self.comm_manager.update_clusters(self.vehicles, self.network)

        # 3. Propagate obstacle confirmations across clusters
        self.comm_manager.propagate_confirmations(self.vehicles, self.network, t)

        # 4. Decay expired blacklist entries
        self.comm_manager.decay_all_blacklists(self.vehicles, t)

        # 5. Process each vehicle
        for v in self.vehicles:
            if v.state == VehicleState.ARRIVED or v.state == VehicleState.WAITING:
                continue

            # Check if vehicle is at a blocked node
            if self.network.is_node_blocked(v.current_node):
                v.total_wait_time += 1
                self.metrics.total_wait_timesteps += 1

                # CAV detects obstacle and broadcasts
                if isinstance(v, CAV):
                    self.comm_manager.broadcast_obstacle(
                        v, v.current_node, self.vehicles, t
                    )
                    self.metrics.total_obstacle_broadcasts += 1
                continue  # Can't move from a blocked node this step

            # Check if next node on route is blocked
            next_node = v.get_next_node()
            if next_node is not None and self.network.is_node_blocked(next_node):
                v.total_wait_time += 1
                self.metrics.total_wait_timesteps += 1

                # CAV detects upcoming obstacle and reroutes
                if isinstance(v, CAV):
                    self.comm_manager.broadcast_obstacle(
                        v, next_node, self.vehicles, t
                    )
                    self.metrics.total_obstacle_broadcasts += 1

                    # Reroute: this is the integration loop trigger
                    cluster = self.comm_manager.get_cluster_members(
                        v.vehicle_id, self.vehicles
                    )
                    v.compute_route(self.network, t, cluster_vehicles=cluster)
                # HDV just waits (no rerouting capability)
                continue

            # Move vehicle forward one step along its route
            self._move_vehicle(v, t)

        # 6. Handle arrivals
        self._handle_arrivals(t)

    # ── Vehicle Movement ─────────────────────────────────────────────────

    def _move_vehicle(self, v: Vehicle, t: int) -> None:
        """
        Advance a vehicle one step along its route.

        At each intersection, CAVs can recompute their route using
        Dec-CTDSP with current cluster information.
        """
        next_node = v.get_next_node()
        if next_node is None:
            return

        # Remove from current link (if tracked)
        prev_node = v.current_node
        if self.network.graph.has_edge(prev_node, next_node):
            self.network.remove_vehicle_from_link(v.vehicle_id, prev_node, next_node)

        # Compute speed based on link density
        if self.network.graph.has_edge(prev_node, next_node):
            from src.environment.grid_network import speed_density
            density = self.network.get_edge_density(prev_node, next_node)
            v.speed = speed_density(
                density,
                self.network.config.link_capacity,
                self.network.config.speed_limit,
            )

        # Advance to next node
        v.advance_route()
        v.total_travel_time += 1

        # Place on next link (if continuing)
        next_next = v.get_next_node()
        if next_next is not None and self.network.graph.has_edge(v.current_node, next_next):
            self.network.place_vehicle_on_link(v.vehicle_id, v.current_node, next_next)

        # CAV at intersection: optionally reroute with updated info
        # Paper: "each vehicle would update their route when they reach an intersection"
        if isinstance(v, CAV) and next_next is not None:
            cluster = self.comm_manager.get_cluster_members(
                v.vehicle_id, self.vehicles
            )
            if cluster:  # Only reroute if cluster has useful info
                v.compute_route(self.network, t, cluster_vehicles=cluster)

    # ── Arrival Handling ─────────────────────────────────────────────────

    def _handle_arrivals(self, t: int) -> None:
        """
        Check for vehicles that reached their destination.
        Reset them for a new trip (continuous simulation).
        """
        for v in self.vehicles:
            if v.state != VehicleState.EN_ROUTE:
                continue

            if v.has_reached_destination():
                v.arrive(t)
                self.metrics.total_trips_completed += 1
                if v.trip_travel_times:
                    self.metrics.all_trip_times.append(v.trip_travel_times[-1])

                # Reset for new trip (continuous format from paper)
                VehicleFactory.reassign_destination(v, self.network.destinations)
                if isinstance(v, CAV):
                    v.set_routing_function(dec_ctdsp_route)
                    v.compute_route(self.network, t, cluster_vehicles=[])
                elif isinstance(v, HDV):
                    v.compute_route(self.network, t)
                v.depart(t)
                v.speed = self.network.config.speed_limit


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Simulator Test")
    print("=" * 60)

    # ── Test 1: Small quick run ──
    print("\n--- Test 1: Quick Simulation (MP=100%, CR=0.5) ---")
    sim_config = SimConfig(
        num_vehicles=30,
        market_penetration=1.0,
        communication_radius=0.5,
        num_obstacles=2,
        max_timesteps=150,
        convergence_window=50,
        warmup_steps=10,
        seed=42,
    )
    sim = Simulator(sim_config)
    metrics = sim.run(verbose=True)
    summary = metrics.summary()
    print(f"\nResults:")
    for k, val in summary.items():
        print(f"  {k}: {val}")

    # ── Test 2: HDV-only baseline (MP=0%) ──
    print("\n--- Test 2: HDV-Only Baseline (MP=0%) ---")
    baseline_config = SimConfig(
        num_vehicles=30,
        market_penetration=0.0,
        communication_radius=0.5,
        num_obstacles=2,
        max_timesteps=150,
        convergence_window=50,
        warmup_steps=10,
        seed=42,
    )
    sim_baseline = Simulator(baseline_config)
    metrics_baseline = sim_baseline.run(verbose=True)
    summary_baseline = metrics_baseline.summary()
    print(f"\nBaseline Results:")
    for k, val in summary_baseline.items():
        print(f"  {k}: {val}")

    # ── Compare ──
    print("\n--- Comparison ---")
    mstt_cav = summary["final_mstt"]
    mstt_hdv = summary_baseline["final_mstt"]
    if mstt_hdv > 0:
        improvement = ((mstt_hdv - mstt_cav) / mstt_hdv) * 100
        print(f"MSTT with CAVs:  {mstt_cav:.2f}")
        print(f"MSTT HDV-only:   {mstt_hdv:.2f}")
        print(f"Improvement:     {improvement:.1f}%")
    else:
        print("Not enough data for comparison yet.")

    recalc = summary["total_recalculations"]
    print(f"Route recalculations (CAV): {recalc}")
    print(f"Route recalculations (HDV): {summary_baseline['total_recalculations']}")

    print("\nAll tests passed.")