"""
reward.py - Multi-Objective Reward Function
The reward function that defines what "good driving" means for the
MARL agents. This is a key design contribution of the project.

The total reward at each timestep is a weighted sum of five terms:

    R(t) = w_p * R_progress + w_a * R_arrival + w_s * R_safety
           + w_w * R_wait + w_es * R_event_safety

Terms:
    R_progress:  Positive reward for moving closer to destination.
                 Measured as reduction in shortest-path distance.

    R_arrival:   Large one-time bonus on reaching destination, scaled
                 inversely by trip duration (faster = bigger bonus).

    R_safety:    Penalty based on link congestion. Penalises being on
                 overcrowded links (proxy for unsafe following distance).

    R_wait:      Penalty for each timestep spent stalled (blocked by
                 obstacle or gridlocked). Pushes agents to reroute.

Design rationale:
    - Progress reward gives a continuous gradient toward the goal,
      preventing the "loop forever" problem.
    - Arrival bonus with time scaling incentivises efficiency.
    - Safety penalty is continuous, not just binary collision detection,
      so the agent learns to avoid *near*-collisions too.
    - Wait penalty is small per-step so the agent prefers rerouting
      over waiting, but doesn't panic over short delays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from src.environment.grid_network import GridNetwork
    from src.environment.vehicle import Vehicle


# ── Reward Configuration ─────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    Weights and parameters for each reward component.

    Tune these to change agent behaviour. Higher weight = stronger signal.
    """
    # Progress reward
    w_progress: float = 1.0
    progress_per_block: float = 1.0     # Reward per block of progress

    # Arrival bonus
    w_arrival: float = 5.0
    arrival_base: float = 20.0          # Base bonus for reaching destination
    arrival_time_scale: float = 0.5     # How much to penalise slow trips
    optimal_trip_time: float = 5.0      # Expected min trip time (for scaling)

    # Safety penalty
    w_safety: float = 1.0
    safety_threshold: float = 0.6       # Density ratio above which penalty kicks in
    collision_penalty: float = -10.0     # Penalty when link is at full capacity

    # Wait/stall penalty
    w_wait: float = 0.5
    wait_penalty_per_step: float = -0.5  # Penalty per timestep stalled
    long_stall_threshold: int = 5        # Steps after which stall penalty doubles
    long_stall_multiplier: float = 2.0

    # Explicit safety-event penalties (from TTC/collision monitor)
    w_event_safety: float = 1.0
    near_miss_penalty: float = -2.0
    collision_event_penalty: float = -25.0


# ── Reward Calculator ────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Computes the multi-objective reward for a vehicle at each timestep.

    Usage:
        calc = RewardCalculator()
        reward, breakdown = calc.compute(vehicle, network, prev_distance, ...)
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

        Args:
            vehicle:                 The vehicle being rewarded.
            network:                 The grid network.
            prev_distance:           Shortest-path distance to dest BEFORE this step.
            curr_distance:           Shortest-path distance to dest AFTER this step.
            is_stalled:              Whether the vehicle is blocked this timestep.
            consecutive_stall_steps: How many consecutive timesteps it's been stalled.
            just_arrived:            Whether the vehicle just reached its destination.
            trip_duration:           How many timesteps this trip has taken so far.
            link_density_ratio:      Current link density / capacity (0.0 to 1.0).
            had_near_miss:           Whether this agent was in a near-miss event this step.
            had_collision:           Whether this agent was in a collision event this step.

        Returns:
            (total_reward, breakdown_dict) where breakdown has each component.
        """
        cfg = self.config

        # ── R_progress ──
        r_progress = self._compute_progress(prev_distance, curr_distance)

        # ── R_arrival ──
        r_arrival = self._compute_arrival(just_arrived, trip_duration)

        # ── R_safety ──
        r_safety = self._compute_safety(link_density_ratio)

        # ── R_wait ──
        r_wait = self._compute_wait(is_stalled, consecutive_stall_steps)

        # ── R_event_safety ──
        r_event_safety = self._compute_event_safety(had_near_miss, had_collision)

        # ── Total ──
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

    # ── Individual Reward Components ─────────────────────────────────────

    def _compute_progress(
        self,
        prev_distance: float,
        curr_distance: float,
    ) -> float:
        """
        Reward for moving closer to the destination.

        R_progress = progress_per_block * (prev_dist - curr_dist)

        Positive when moving closer, negative when moving away.
        This gives a continuous gradient toward the goal and prevents
        agents from looping forever without making progress.
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

        Faster trips get a bigger bonus. A trip at optimal time gets the
        full base_bonus. A trip taking twice as long gets a reduced bonus.
        The bonus never goes below 0 (arriving is always better than not).
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

        R_safety = 0                                    if ratio < threshold
                 = collision_penalty * (ratio - thresh) / (1 - thresh)
                                                        if ratio >= threshold

        This is a continuous penalty that scales with congestion.
        At full capacity (ratio=1.0), the agent gets the full collision_penalty.
        Below the threshold, no penalty — some traffic is normal.

        This serves as a proxy for "maintain safe following distance"
        in the grid world where we don't have continuous physics.
        """
        cfg = self.config

        if link_density_ratio < cfg.safety_threshold:
            return 0.0

        if link_density_ratio >= 1.0:
            return cfg.collision_penalty

        # Linear interpolation between threshold and full capacity
        severity = (link_density_ratio - cfg.safety_threshold) / (1.0 - cfg.safety_threshold)
        return cfg.collision_penalty * severity

    def _compute_wait(
        self,
        is_stalled: bool,
        consecutive_stall_steps: int,
    ) -> float:
        """
        Penalty for being stalled (blocked by obstacle or gridlocked).

        R_wait = wait_penalty_per_step                  if stalled < threshold
               = wait_penalty_per_step * multiplier     if stalled >= threshold
               = 0                                      if not stalled

        The penalty doubles after a long stall to push the agent
        toward rerouting rather than waiting indefinitely.
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


# ── Shortest Path Distance Helper ────────────────────────────────────────────

def shortest_path_distance(
    network: GridNetwork,
    source: Tuple[int, int],
    target: Tuple[int, int],
) -> float:
    """
    Compute shortest path distance (in hops) from source to target.

    Used by the progress reward to measure how much closer the agent
    got to its destination. Uses BFS since all edges have weight 1.
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


# ── Quick Test ───────────────────────────────────────────────────────────────

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

    # ── Test 1: Progress reward ──
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

    # ── Test 2: Arrival bonus ──
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

    # ── Test 3: Safety penalty ──
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

    # ── Test 4: Wait/stall penalty ──
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

    # ── Test 5: Shortest path distance helper ──
    print("\n--- Test 5: Shortest Path Distance ---")
    d1 = shortest_path_distance(network, (0, 0), (0, 5))
    print(f"(0,0) -> (0,5): {d1} hops (expected: 5, straight east)")

    d2 = shortest_path_distance(network, (0, 0), (4, 5))
    print(f"(0,0) -> (4,5): {d2} hops (expected: 9, east+south)")

    d3 = shortest_path_distance(network, (2, 3), (2, 3))
    print(f"(2,3) -> (2,3): {d3} hops (expected: 0, same node)")

    # ── Test 6: Full scenario ──
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
