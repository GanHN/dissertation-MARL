"""
safety.py - Safety event detection utilities
Adds lightweight safety metrics:
    - near-miss events (TTC below threshold)
    - collision events (very small headway with closing speed)
The current traffic model is link-based and continuous along edges.
So safety checks are evaluated on vehicles sharing the same directed link.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.environment.vehicle import Vehicle, VehicleState


@dataclass
class SafetyConfig:
    """Tuneable thresholds for safety event detection."""
    ttc_near_miss_threshold: float = 2.0
    collision_gap_threshold_blocks: float = 0.02
    collision_rel_speed_threshold: float = 0.05


@dataclass
class SafetyStepStats:
    """Safety events detected during one simulation timestep."""
    near_misses: int = 0
    collisions: int = 0
    near_miss_vehicle_ids: Set[int] = field(default_factory=set)
    collision_vehicle_ids: Set[int] = field(default_factory=set)


class SafetyMonitor:
    """
    Detect safety events from current vehicle states.

    Logic (same-link following model):
        - Sort vehicles on each directed link by progress.
        - Evaluate each leader-follower adjacent pair.
        - Near miss if TTC < threshold.
        - Collision if gap is tiny and follower is still closing.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()

    def evaluate(self, vehicles: List[Vehicle]) -> SafetyStepStats:
        cfg = self.config
        by_link: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Vehicle]] = {}

        for v in vehicles:
            if v.state != VehicleState.EN_ROUTE:
                continue
            if v.link_start_node is None or v.link_end_node is None:
                continue
            key = (v.link_start_node, v.link_end_node)
            by_link.setdefault(key, []).append(v)

        stats = SafetyStepStats()

        for link_vehicles in by_link.values():
            if len(link_vehicles) < 2:
                continue

            # Front-most first (higher progress is closer to downstream node)
            ordered = sorted(link_vehicles, key=lambda x: x.link_progress, reverse=True)

            for i in range(len(ordered) - 1):
                leader = ordered[i]
                follower = ordered[i + 1]

                gap = max(0.0, leader.link_progress - follower.link_progress)
                rel_speed = follower.speed - leader.speed
                if rel_speed <= 1e-8:
                    continue

                ttc = gap / rel_speed
                if ttc < cfg.ttc_near_miss_threshold:
                    stats.near_misses += 1
                    stats.near_miss_vehicle_ids.add(leader.vehicle_id)
                    stats.near_miss_vehicle_ids.add(follower.vehicle_id)

                if (
                    gap <= cfg.collision_gap_threshold_blocks
                    and rel_speed >= cfg.collision_rel_speed_threshold
                ):
                    stats.collisions += 1
                    stats.collision_vehicle_ids.add(leader.vehicle_id)
                    stats.collision_vehicle_ids.add(follower.vehicle_id)

        return stats
