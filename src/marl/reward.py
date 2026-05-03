"""
reward.py - Multi-Objective Reward Function
The reward function that defines what "good driving" means for the
MARL agents. This is a key design contribution of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from src.environment.grid_network import GridNetwork
    from src.environment.vehicle import Vehicle



@dataclass
class RewardConfig:
    """
    Weights and parameters for each reward component.
    Higher weight = stronger signal.
    """
    # Progress reward
    w_progress: float = 1.0
    progress_per_block: float = 1.0     # Reward per block of progress

    # Arrival bonus
    w_arrival: float = 3.0              # was 5.0,
    arrival_base: float = 15.0          # Base bonus for reaching destination   # was 20.0
    arrival_time_scale: float = 0.5     # How much to penalise slow trips
    optimal_trip_time: float = 5.0      # Expected min trip time (for scaling)

    # Safety penalty
    w_safety: float = 1.0
    safety_threshold: float = 0.6       # Density ratio above which penalty kicks in
    congestion_penalty: float = -10.0     # Penalty when link is at full capacity

    # stalling penalty
    w_wait: float = 0.5
    wait_penalty_per_step: float = -0.5  # Penalty per timestep stalled
    long_stall_threshold: int = 5        # Steps after which stall penalty doubles
    long_stall_multiplier: float = 2.0

    # Explicit safety-event penalties
    w_event_safety: float = 1.0
    near_miss_penalty: float = -12.0          # was -2.0, increased to make it more significant
    collision_event_penalty: float = -75.0   # was -25.0



class RewardCalculator:
    """
    Computes the multi-objective reward for a vehicle at each timestep.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        vehicle: Vehicle,
        network: GridNetwork,
        prev_distance: float,
        curr_distance: float,
        is_stalled: bool,
        consecutive_stall_steps: int,
        just_arrived: bool,
        trip_duration: float,
        link_density_ratio: float,
        had_near_miss: bool = False,
        had_collision: bool = False,
    ) -> Tuple[float, dict]:
        """
        Compute the total reward for a vehicle at this timestep.
        """
        cfg = self.config

        # R_progress
        r_progress = self._compute_progress(prev_distance, curr_distance)

        # R_arrival
        r_arrival = self._compute_arrival(just_arrived, trip_duration)

        # R_safety
        r_safety = self._compute_safety(link_density_ratio)

        # R_wait
        r_wait = self._compute_wait(is_stalled, consecutive_stall_steps)

        # R_event_safety
        r_event_safety = self._compute_event_safety(had_near_miss, had_collision)

        # Total
        total = (
            cfg.w_progress * r_progress
            + cfg.w_arrival * r_arrival
            + cfg.w_safety * r_safety
            + cfg.w_wait * r_wait
            + cfg.w_event_safety * r_event_safety
        )

        breakdown = {
            "r_progress": round(r_progress, 4),
            "r_arrival": round(r_arrival, 4),
            "r_safety": round(r_safety, 4),
            "r_wait": round(r_wait, 4),
            "r_event_safety": round(r_event_safety, 4),
            "total": round(total, 4),
            "w_progress": round(cfg.w_progress * r_progress, 4),
            "w_arrival": round(cfg.w_arrival * r_arrival, 4),
            "w_safety": round(cfg.w_safety * r_safety, 4),
            "w_wait": round(cfg.w_wait * r_wait, 4),
            "w_event_safety": round(cfg.w_event_safety * r_event_safety, 4),
        }

        return total, breakdown


    def _compute_progress(
        self,
        prev_distance: float,
        curr_distance: float,
    ) -> float:
        """
        Reward for moving closer to the destination.
        """
        delta = prev_distance - curr_distance
        return self.config.progress_per_block * delta

    def _compute_arrival(
        self,
        just_arrived: bool,
        trip_duration: float,
    ) -> float:
        """
        One-time bonus for reaching the destination.
        R_arrival = base_bonus * max(0, 1 - time_scale * (duration - optimal) / optimal)
        """
        if not just_arrived:
            return 0.0

        cfg = self.config
        if cfg.optimal_trip_time <= 0:
            return cfg.arrival_base

        # Scale factor: 1.0 at optimal time, decreasing for slower trips
        time_ratio = (trip_duration - cfg.optimal_trip_time) / cfg.optimal_trip_time
        scale = max(0.0, 1.0 - cfg.arrival_time_scale * time_ratio)

        return cfg.arrival_base * scale

    def _compute_safety(
        self,
        link_density_ratio: float,
    ) -> float:
        """
        Penalty for being on congested/overcrowded links.
        """
        cfg = self.config

        if link_density_ratio < cfg.safety_threshold:
            return 0.0

        if link_density_ratio >= 1.0:
            return cfg.congestion_penalty

        # Linear interpolation between threshold and full capacity
        severity = (link_density_ratio - cfg.safety_threshold) / (1.0 - cfg.safety_threshold)
        return cfg.congestion_penalty * severity

    def _compute_wait(
        self,
        is_stalled: bool,
        consecutive_stall_steps: int,
    ) -> float:
        """
        Penalty for being stalled (blocked by obstacle or gridlocked).
        """
        if not is_stalled:
            return 0.0

        cfg = self.config
        if consecutive_stall_steps >= cfg.long_stall_threshold:
            return cfg.wait_penalty_per_step * cfg.long_stall_multiplier
        return cfg.wait_penalty_per_step

    def _compute_event_safety(
        self,
        had_near_miss: bool,
        had_collision: bool,
    ) -> float:
        """
        Explicit event-based safety penalty.
        Collision is treated as more severe than near miss.
        """
        cfg = self.config
        if had_collision:
            return cfg.collision_event_penalty
        if had_near_miss:
            return cfg.near_miss_penalty
        return 0.0



def shortest_path_distance(
    network: GridNetwork,
    source: Tuple[int, int],
    target: Tuple[int, int],
) -> float:
    """
    Compute shortest path distance (in hops) from source to target.
    """
    if source == target:
        return 0.0

    from collections import deque

    visited = {source}
    queue = deque([(source, 0)])

    while queue:
        node, dist = queue.popleft()
        for neighbor in network.get_neighbors(node):
            if neighbor == target:
                return float(dist + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    # No path found — return large distance
    return 100.0



if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.environment.grid_network import GridNetwork, NetworkConfig
    from src.environment.vehicle import CAV

    network = GridNetwork(NetworkConfig())
    calc = RewardCalculator()

    print("=" * 60)
    print("Reward Function Test")
    print("=" * 60)

    #Test 1: Progress reward
    print("\n--- Test 1: Progress Reward ---")
    reward, bd = calc.compute(
        vehicle=None, network=network,
        prev_distance=8.0, curr_distance=7.0,
        is_stalled=False, consecutive_stall_steps=0,
        just_arrived=False, trip_duration=3,
        link_density_ratio=0.2,
    )
    print(f"Moved 1 block closer: reward={reward:.3f}")
    print(f"  Breakdown: {bd}")

    # Moving away
    reward2, bd2 = calc.compute(
        vehicle=None, network=network,
        prev_distance=7.0, curr_distance=8.0,
        is_stalled=False, consecutive_stall_steps=0,
        just_arrived=False, trip_duration=4,
        link_density_ratio=0.2,
    )
    print(f"Moved 1 block AWAY: reward={reward2:.3f}")
    print(f"  Progress component: {bd2['r_progress']}")

    # Test 2: Arrival bonus
    print("\n--- Test 2: Arrival Bonus ---")
    # Fast arrival (at optimal time)
    reward_fast, bd_fast = calc.compute(
        vehicle=None, network=network,
        prev_distance=1.0, curr_distance=0.0,
        is_stalled=False, consecutive_stall_steps=0,
        just_arrived=True, trip_duration=5.0,
        link_density_ratio=0.1,
    )
    print(f"Fast arrival (5 steps): reward={reward_fast:.3f}, arrival_bonus={bd_fast['r_arrival']}")

    # Slow arrival
    reward_slow, bd_slow = calc.compute(
        vehicle=None, network=network,
        prev_distance=1.0, curr_distance=0.0,
        is_stalled=False, consecutive_stall_steps=0,
        just_arrived=True, trip_duration=15.0,
        link_density_ratio=0.1,
    )
    print(f"Slow arrival (15 steps): reward={reward_slow:.3f}, arrival_bonus={bd_slow['r_arrival']}")
    print(f"  Fast > Slow? {reward_fast > reward_slow}")

    # Test 3: Safety penalty
    print("\n--- Test 3: Safety Penalty ---")
    for ratio in [0.0, 0.3, 0.6, 0.8, 1.0]:
        _, bd_s = calc.compute(
            vehicle=None, network=network,
            prev_distance=5.0, curr_distance=5.0,
            is_stalled=False, consecutive_stall_steps=0,
            just_arrived=False, trip_duration=3,
            link_density_ratio=ratio,
        )
        print(f"  Density ratio={ratio:.1f}: safety_penalty={bd_s['r_safety']:.3f}")

    # Test 4: Wait/stall penalty
    print("\n--- Test 4: Wait/Stall Penalty ---")
    for stall_steps in [0, 1, 3, 5, 10]:
        is_stalled = stall_steps > 0
        _, bd_w = calc.compute(
            vehicle=None, network=network,
            prev_distance=5.0, curr_distance=5.0,
            is_stalled=is_stalled, consecutive_stall_steps=stall_steps,
            just_arrived=False, trip_duration=3,
            link_density_ratio=0.2,
        )
        print(f"  Stall steps={stall_steps:2d}: wait_penalty={bd_w['r_wait']:.3f}")

    # Test 5: Shortest path distance helper
    print("\n--- Test 5: Shortest Path Distance ---")
    d1 = shortest_path_distance(network, (0, 0), (0, 5))
    print(f"(0,0) -> (0,5): {d1} hops (expected: 5, straight east)")

    d2 = shortest_path_distance(network, (0, 0), (4, 5))
    print(f"(0,0) -> (4,5): {d2} hops (expected: 9, east+south)")

    d3 = shortest_path_distance(network, (2, 3), (2, 3))
    print(f"(2,3) -> (2,3): {d3} hops (expected: 0, same node)")

    # Test 6: Full scenario
    print("\n--- Test 6: Full Trip Scenario ---")
    print("Simulating a 5-step trip with varying conditions:")
    scenarios = [
        {"prev": 5.0, "curr": 4.0, "stalled": False, "stall_n": 0, "arrived": False, "dur": 1, "density": 0.2},
        {"prev": 4.0, "curr": 3.0, "stalled": False, "stall_n": 0, "arrived": False, "dur": 2, "density": 0.4},
        {"prev": 3.0, "curr": 3.0, "stalled": True,  "stall_n": 1, "arrived": False, "dur": 3, "density": 0.8},
        {"prev": 3.0, "curr": 2.0, "stalled": False, "stall_n": 0, "arrived": False, "dur": 4, "density": 0.3},
        {"prev": 2.0, "curr": 0.0, "stalled": False, "stall_n": 0, "arrived": True,  "dur": 5, "density": 0.1},
    ]

    total_episode_reward = 0.0
    for i, s in enumerate(scenarios):
        r, bd = calc.compute(
            vehicle=None, network=network,
            prev_distance=s["prev"], curr_distance=s["curr"],
            is_stalled=s["stalled"], consecutive_stall_steps=s["stall_n"],
            just_arrived=s["arrived"], trip_duration=s["dur"],
            link_density_ratio=s["density"],
        )
        total_episode_reward += r
        status = "ARRIVED" if s["arrived"] else ("STALLED" if s["stalled"] else "moving")
        print(f"  Step {i+1}: {status:8s} | reward={r:+7.3f} | "
              f"prog={bd['r_progress']:+.2f} arr={bd['r_arrival']:+.2f} "
              f"safe={bd['r_safety']:+.2f} wait={bd['r_wait']:+.2f}")

    print(f"  Total episode reward: {total_episode_reward:.3f}")

    print("\nAll tests passed.")
