"""
benchmark.py - Comprehensive System Benchmarking
Compares full Dec-CTDSP + OMM + MA2C system against ablations and baselines to prove each component adds measurable value.
Configurations tested:
    1. HDV-only (MP=0%)             - No CAVs, pure HDV routing
    2. Static Dijkstra (no OMM)     - CAVs use Dijkstra but no obstacle memory
    3. Dec-CTDSP + OMM              - Full high-level routing system
    4. Dec-CTDSP + MA2C (no OMM)     - CAVs use MA2C for decisions but no OMM
    5. Dec-CTDSP + OMM + MA2C       - Full system
Metrics compared:
    - MSTT (Mean System Travel Time)
    - MSS (Mean System Speed)
    - Average wait time per trip (proves OMM value)
    - Total trips completed (throughput)
    - Standard deviation of MSTT (reliability)
    - Number of stalled vehicles
    - Collision rates **
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_network import GridNetwork, NetworkConfig
from src.environment.vehicle import (
    CAV, HDV, Vehicle, VehicleFactory, VehicleState, VehicleType,
)
from src.communication.comm_manager import CommunicationManager, CommConfig
from src.routing.dec_ctdsp import dec_ctdsp_route
from src.environment.simulator import Simulator, SimConfig, ObstacleManager
from configs.experiment_defaults import BLACKLIST_TTL, COMM_RADIUS, NUM_OBSTACLES
from src.marl.ma2c import MA2CAgent, MA2CConfig
from src.marl.gat_network import GATConfig

import torch



CONFIG_LABELS = {
    "hdv_only": "HDV only\n(baseline)",
    "static_dijkstra": "Static Dijkstra",
    "dec_ctdsp": "Dec-CTDSP\n+ OMM",
    "dec_ctdsp_omm_no_decay": "Dec-CTDSP\n+ OMM\n(no TTL decay)",
    "dec_ctdsp_ma2c_no_omm": "Dec-CTDSP\n+ MA2C\n(no OMM)",
    "dec_ctdsp_rule_heuristic": "Dec-CTDSP\n+ OMM + Rule\n(heuristic)",
    "dec_ctdsp_ma2c_no_gat": "Dec-CTDSP\n+ OMM + MA2C\n(no GAT)",
    "dec_ctdsp_marl": "Dec-CTDSP\n+ OMM + MA2C\n(full system)",
}

CONFIG_COLORS = {
    "hdv_only": "#94A3B8",              
    "static_dijkstra": "#F59E0B",       
    "dec_ctdsp": "#3B82F6",             
    "dec_ctdsp_omm_no_decay": "#2563EB",
    "dec_ctdsp_ma2c_no_omm": "#8B5CF6", 
    "dec_ctdsp_rule_heuristic": "#0EA5A4",
    "dec_ctdsp_ma2c_no_gat": "#6366F1",
    "dec_ctdsp_marl": "#10B981",        
}

CONFIG_ORDER = [
    "hdv_only",
    "static_dijkstra",
    "dec_ctdsp",
    "dec_ctdsp_omm_no_decay",
    "dec_ctdsp_ma2c_no_omm",
    "dec_ctdsp_rule_heuristic",
    "dec_ctdsp_ma2c_no_gat",
    "dec_ctdsp_marl",
]

COMPARISON_ORDER = [cfg for cfg in CONFIG_ORDER if cfg != "hdv_only"]



def run_hdv_only(config: SimConfig, seed: int) -> Dict:
    """Run baseline with 0% market penetration (all HDVs)."""
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 0.0, "seed": seed})
    sim = Simulator(cfg)
    metrics = sim.run(verbose=False)
    summary = metrics.summary()
    summary["config_name"] = "hdv_only"
    summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
    return summary


def run_static_dijkstra(config: SimConfig, seed: int) -> Dict:
    """
    Run with CAVs but WITHOUT Dec-CTDSP or OMM.
    CAVs use simple static Dijkstra (from vehicle.py fallback)
    and their OMM blacklists are cleared each timestep, so they have no obstacle memory.
    """
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 1.0, "seed": seed})
    sim = Simulator(cfg)

    # Disable Dec-CTDSP routing and OMM by not injecting the routing function
    # and forcing blacklists to clear each step
    for v in sim.vehicles:
        if isinstance(v, CAV):
            v._routing_function = None  # Forces fallback Dijkstra

    # Monkey-patch the simulator to clear blacklists each step
    original_step = sim._step
    def patched_step(t):
        for v in sim.vehicles:
            if isinstance(v, CAV):
                v.blacklist.clear()
        original_step(t)
    sim._step = patched_step

    metrics = sim.run(verbose=False)
    summary = metrics.summary()
    summary["config_name"] = "static_dijkstra"
    summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
    return summary


def run_dec_ctdsp(config: SimConfig, seed: int) -> Dict:
    """Run full Dec-CTDSP + OMM system (no MA2C)."""
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 1.0, "seed": seed})
    sim = Simulator(cfg)
    metrics = sim.run(verbose=False)
    summary = metrics.summary()
    summary["config_name"] = "dec_ctdsp"
    summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
    return summary


def run_dec_ctdsp_omm_no_decay(config: SimConfig, seed: int) -> Dict:
    """
    Run Dec-CTDSP + OMM without TTL-based decay (original-style persistent OMM).

    Practical implementation:
    - Use a very large blacklist TTL so per-vehicle local decay never expires entries
      within benchmark horizon.
    - Disable global periodic decay refresh/removal pass.
    """
    cfg = SimConfig(
        **{
            **config.__dict__,
            "market_penetration": 1.0,
            "seed": seed,
            "blacklist_ttl": 10**9,
        }
    )
    sim = Simulator(cfg)
    sim.comm_manager.decay_all_blacklists = lambda *args, **kwargs: {}
    metrics = sim.run(verbose=False)
    summary = metrics.summary()
    summary["config_name"] = "dec_ctdsp_omm_no_decay"
    summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
    return summary


def run_dec_ctdsp_marl(
    config: SimConfig,
    seed: int,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Run full system including trained MA2C agent making action decisions.
    Uses the MARL environment wrapper so the agent actually picks actions
    (follow/alternative/wait/reroute) rather than just running autonomously.
    Falls back to the autonomous simulator if no model is available.
    """
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 1.0, "seed": seed})

    if not (model_path and os.path.exists(model_path)):
        # No model — fall back to autonomous simulator (same as Dec-CTDSP+OMM)
        sim = Simulator(cfg)
        metrics = sim.run(verbose=False)
        summary = metrics.summary()
        summary["config_name"] = "dec_ctdsp_marl"
        summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
        summary["marl_model_used"] = False
        return summary

    # Use the trained agent via the MARL environment
    return _run_with_trained_agent(
        cfg, seed, model_path, config_name="dec_ctdsp_marl", enable_omm=True,
    )


def run_dec_ctdsp_ma2c_no_omm(
    config: SimConfig,
    seed: int,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Run Dec-CTDSP + MA2C but WITHOUT OMM.
    CAVs use Dec-CTDSP for routing and the MA2C agent for decisions,
    but their blacklists are cleared every timestep so there's no
    persistent obstacle memory. This ablation proves whether OMM
    actually contributes to performance vs MARL alone.
    """
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 1.0, "seed": seed})

    if not (model_path and os.path.exists(model_path)):
        # No model — fall back to simulator with blacklists cleared
        sim = Simulator(cfg)
        sim.comm_manager.broadcast_obstacle = lambda *args, **kwargs: 0
        sim.comm_manager.propagate_confirmations = lambda *args, **kwargs: 0
        sim.comm_manager.decay_all_blacklists = lambda *args, **kwargs: {}
        original_step = sim._step
        def patched_step(t):
            for v in sim.vehicles:
                if isinstance(v, CAV):
                    v.blacklist.clear()
            original_step(t)
        sim._step = patched_step
        metrics = sim.run(verbose=False)
        summary = metrics.summary()
        summary["config_name"] = "dec_ctdsp_ma2c_no_omm"
        summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
        summary["marl_model_used"] = False
        return summary

    # Use the trained agent with OMM disabled
    return _run_with_trained_agent(
        cfg, seed, model_path, config_name="dec_ctdsp_ma2c_no_omm", enable_omm=False,
    )


def run_dec_ctdsp_rule_heuristic(
    config: SimConfig,
    seed: int,
) -> Dict:
    """
    Run Dec-CTDSP + OMM with a fixed rule-based policy instead of MA2C.

    This isolates whether the learned policy adds value beyond a simple
    handcrafted decision policy over the same action space.
    """
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 1.0, "seed": seed})
    return _run_with_trained_agent(
        cfg,
        seed,
        model_path=None,
        config_name="dec_ctdsp_rule_heuristic",
        enable_omm=True,
        use_gat_context=False,
        use_rule_policy=True,
    )


def run_dec_ctdsp_ma2c_no_gat(
    config: SimConfig,
    seed: int,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Run Dec-CTDSP + OMM + MA2C with GAT context disabled at execution time.

    This isolates the effect of inter-agent context aggregation vs.
    independent local decision-making.
    """
    cfg = SimConfig(**{**config.__dict__, "market_penetration": 1.0, "seed": seed})
    if not (model_path and os.path.exists(model_path)):
        sim = Simulator(cfg)
        metrics = sim.run(verbose=False)
        summary = metrics.summary()
        summary["config_name"] = "dec_ctdsp_ma2c_no_gat"
        summary["mstt_std"] = float(np.std(metrics.mstt_history[-100:])) if len(metrics.mstt_history) >= 100 else 0.0
        summary["marl_model_used"] = False
        return summary

    return _run_with_trained_agent(
        cfg,
        seed,
        model_path=model_path,
        config_name="dec_ctdsp_ma2c_no_gat",
        enable_omm=True,
        use_gat_context=False,
        use_rule_policy=False,
    )


def _rule_action_for_cav(env, cav: CAV) -> int:
    """
    Handcrafted policy over the 4-action space:
      0 follow, 1 alternative, 2 wait, 3 reroute
    """
    if cav.state != VehicleState.EN_ROUTE:
        return 0
    if not cav.is_at_intersection():
        return 0

    next_node = cav.get_next_node()
    if next_node is None:
        return 3
    if env.network.is_node_blocked(next_node):
        return 3

    blacklisted = cav.get_blacklisted_nodes()
    remaining = cav.get_remaining_route()
    if any(node in blacklisted for node in remaining):
        return 3

    stall_steps = env._stall_counters.get(cav.vehicle_id, 0)
    if stall_steps >= 3:
        return 3

    density_ratio = 0.0
    from_n = cav.current_node
    to_n = next_node
    if env.network.graph.has_edge(from_n, to_n):
        density = env.network.get_edge_density(from_n, to_n)
        capacity = env.network.graph.edges[from_n, to_n]["capacity"]
        density_ratio = density / max(1, capacity)

    if density_ratio >= 0.95 and stall_steps >= 1:
        return 2
    if density_ratio >= 0.80:
        return 1
    return 0


def _run_with_trained_agent(
    cfg: SimConfig,
    seed: int,
    model_path: Optional[str],
    config_name: str,
    enable_omm: bool,
    use_gat_context: bool = True,
    use_rule_policy: bool = False,
    omm_no_decay: bool = False,
) -> Dict:
    """
    runs a simulation where the trained MA2C agent
    actually picks actions via the MARL environment.
    Tracks the same metrics as the autonomous Simulator so results
    are comparable across configs.
    """
    # Lazy import to avoid circular dependency
    from src.train_marl import MARLEnvironment, TrainConfig

    # Build training config from sim config
    tc = TrainConfig()
    tc.num_vehicles = cfg.num_vehicles
    tc.market_penetration = cfg.market_penetration
    tc.communication_radius = cfg.communication_radius
    tc.num_obstacles = cfg.num_obstacles
    tc.blacklist_ttl = 10**9 if omm_no_decay else cfg.blacklist_ttl
    tc.grid_rows = cfg.grid_rows
    tc.grid_cols = cfg.grid_cols
    tc.steps_per_episode = cfg.max_timesteps
    tc.seed = seed

    # Load trained agent (not required for pure heuristic mode)
    agent: Optional[MA2CAgent] = None
    if not use_rule_policy:
        if not (model_path and os.path.exists(model_path)):
            raise FileNotFoundError(
                f"Model path required for {config_name}: {model_path}"
            )
        agent = MA2CAgent()
        agent.load(model_path)
        agent.set_eval_mode()

    # Build environment
    env = MARLEnvironment(tc)
    env.reset(seed=seed)
    if not enable_omm:
        env.comm.broadcast_obstacle = lambda *args, **kwargs: 0
        env.comm.propagate_confirmations = lambda *args, **kwargs: 0
        env.comm.decay_all_blacklists = lambda *args, **kwargs: {}
    elif omm_no_decay:
        env.comm.decay_all_blacklists = lambda *args, **kwargs: {}

    # Run the episode, tracking metrics manually
    all_trip_times = []
    total_recalcs = 0
    total_broadcasts = 0
    total_wait_steps = 0
    total_trips = 0
    total_near_misses = 0
    total_collisions = 0
    total_vehicle_steps = 0
    mstt_history = []
    speeds_history = []
    prev_total_wait = 0

    for step in range(cfg.max_timesteps):
        # Optionally clear blacklists (disables OMM)
        if not enable_omm:
            for v in env.vehicles:
                if isinstance(v, CAV):
                    v.blacklist.clear()
        # Count vehicle-steps at timestep start (aligned with Simulator logic).
        active_now = sum(1 for v in env.vehicles if v.state == VehicleState.EN_ROUTE)
        total_vehicle_steps += active_now

        # Get observations
        node_features, edge_index, vid_order = env.get_gat_inputs()
        global_state = env.get_global_state()

        if len(vid_order) == 0:
            # Preserve fixed-horizon fairness even if all CAVs become inactive.
            env.step({})
            total_trips = sum(v.trips_completed for v in env.vehicles)
            total_recalcs = sum(v.num_route_recalculations for v in env.vehicles if isinstance(v, CAV))
            total_broadcasts = env.comm.obstacle_broadcasts_sent
            total_near_misses = env.total_near_misses
            total_collisions = env.total_collisions

            if step >= cfg.warmup_steps:
                recent = [v.trip_travel_times[-1] for v in env.vehicles if v.trip_travel_times]
                if recent:
                    mstt_history.append(np.mean(recent))
                active_speeds = [v.speed for v in env.vehicles
                                 if v.state == VehicleState.EN_ROUTE and v.speed > 0]
                if active_speeds:
                    speeds_history.append(np.mean(active_speeds))

            curr_total_wait = sum(v.total_wait_time for v in env.vehicles)
            step_wait = max(0, curr_total_wait - prev_total_wait)
            total_wait_steps += step_wait
            prev_total_wait = curr_total_wait
            continue

        actions: Dict[int, int] = {}
        if use_rule_policy:
            cav_by_id = {cav.vehicle_id: cav for cav in env.cavs}
            for vid in vid_order:
                cav = cav_by_id.get(vid)
                actions[vid] = _rule_action_for_cav(env, cav) if cav else 0
        else:
            assert agent is not None
            # MA2C action selection, with optional GAT context disabled.
            if use_gat_context:
                with torch.no_grad():
                    contexts = agent.gat(node_features, edge_index)
            else:
                contexts = torch.zeros(
                    (len(vid_order), agent.config.context_dim),
                    dtype=torch.float32,
                )

            vid_to_idx = {vid: i for i, vid in enumerate(vid_order)}
            local_obs_map = env.get_observations()

            for vid in vid_order:
                idx = vid_to_idx[vid]
                local_obs = local_obs_map.get(vid)
                if local_obs is None:
                    actions[vid] = 0
                    continue
                context = contexts[idx]
                action, _, _ = agent.act(
                    local_obs, context, global_state, deterministic=True
                )
                actions[vid] = action

        # Step environment
        env.step(actions)

        # Accumulate metrics
        # Trips completed this step
        step_trips = sum(v.trips_completed for v in env.vehicles)
        total_trips = step_trips
        total_recalcs = sum(v.num_route_recalculations for v in env.vehicles if isinstance(v, CAV))
        total_broadcasts = env.comm.obstacle_broadcasts_sent
        total_near_misses = env.total_near_misses
        total_collisions = env.total_collisions

        # Align MARL metric recording with Simulator behavior:
        # only record MSTT/MSS after warmup_steps.
        if step >= cfg.warmup_steps:
            # Recent trip times for MSTT
            recent = []
            for v in env.vehicles:
                if v.trip_travel_times:
                    recent.append(v.trip_travel_times[-1])
            if recent:
                mstt_history.append(np.mean(recent))

            # Speeds
            active_speeds = [v.speed for v in env.vehicles
                             if v.state == VehicleState.EN_ROUTE and v.speed > 0]
            if active_speeds:
                speeds_history.append(np.mean(active_speeds))

        # Wait tracking: system-wide cumulative wait across all vehicles
        # (CAV + HDV), aligned with Simulator-based configs.
        curr_total_wait = sum(v.total_wait_time for v in env.vehicles)
        step_wait = max(0, curr_total_wait - prev_total_wait)
        total_wait_steps += step_wait
        prev_total_wait = curr_total_wait

    # Compute final summary
    final_mstt = float(np.mean(mstt_history[-100:])) if len(mstt_history) >= 10 else (
        float(np.mean(mstt_history)) if mstt_history else 0.0
    )
    final_mss = float(np.mean(speeds_history[-100:])) if len(speeds_history) >= 10 else (
        float(np.mean(speeds_history)) if speeds_history else 0.0
    )

    # Collect all trip times for avg_trip_time
    for v in env.vehicles:
        all_trip_times.extend(v.trip_travel_times)

    return {
        "config_name": config_name,
        "final_mstt": round(final_mstt, 3),
        "final_mss": round(final_mss, 3),
        "total_trips": total_trips,
        "total_recalculations": total_recalcs,
        "total_obstacle_broadcasts": total_broadcasts,
        "avg_trip_time": round(float(np.mean(all_trip_times)), 3) if all_trip_times else 0,
        "avg_wait_per_vehicle": round(total_wait_steps / max(1, total_trips), 3),
        "total_near_misses": total_near_misses,
        "total_collisions": total_collisions,
        "near_miss_rate_per_1000_trips": round((total_near_misses * 1000.0) / max(1, total_trips), 3),
        "collision_rate_per_1000_trips": round((total_collisions * 1000.0) / max(1, total_trips), 3),
        "near_miss_rate_per_10k_vehicle_steps": round((total_near_misses * 10000.0) / max(1, total_vehicle_steps), 3),
        "collision_rate_per_10k_vehicle_steps": round((total_collisions * 10000.0) / max(1, total_vehicle_steps), 3),
        "timesteps_run": len(mstt_history),
        "mstt_std": float(np.std(mstt_history[-100:])) if len(mstt_history) >= 100 else 0.0,
        "marl_model_used": not use_rule_policy,
        "uses_gat_context": use_gat_context and not use_rule_policy,
    }



def run_benchmark(
    num_vehicles: int = 100,
    num_seeds: int = 3,
    max_timesteps: int = 500,
    num_obstacles: int = NUM_OBSTACLES,
    cr: float = COMM_RADIUS,
    model_path: Optional[str] = None,
    model_path_no_omm: Optional[str] = None,
    model_path_no_gat: Optional[str] = None,
    output_dir: str = "results/benchmark",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all benchmark configurations and return aggregated results."""
    os.makedirs(output_dir, exist_ok=True)

    base_config = SimConfig(
        num_vehicles=num_vehicles,
        communication_radius=cr,
        num_obstacles=num_obstacles,
        blacklist_ttl=BLACKLIST_TTL,
        max_timesteps=max_timesteps,
        # Force a fixed-horizon benchmark across all configs.
        # Simulator-based configs would otherwise stop early on convergence,
        # while MARL rollouts run the full max_timesteps.
        convergence_window=max_timesteps + 1,
        warmup_steps=10,
    )

    all_results = []
    no_omm_model = model_path_no_omm or model_path
    no_gat_model = model_path_no_gat or model_path

    configs = [
        ("hdv_only", run_hdv_only),
        ("static_dijkstra", run_static_dijkstra),
        ("dec_ctdsp", run_dec_ctdsp),
        ("dec_ctdsp_omm_no_decay", run_dec_ctdsp_omm_no_decay),
        ("dec_ctdsp_ma2c_no_omm", lambda c, s: run_dec_ctdsp_ma2c_no_omm(c, s, no_omm_model)),
        ("dec_ctdsp_rule_heuristic", lambda c, s: run_dec_ctdsp_rule_heuristic(c, s)),
        ("dec_ctdsp_ma2c_no_gat", lambda c, s: run_dec_ctdsp_ma2c_no_gat(c, s, no_gat_model)),
        ("dec_ctdsp_marl", lambda c, s: run_dec_ctdsp_marl(c, s, model_path)),
    ]

    if verbose:
        print("=" * 60)
        print("Benchmark Runner")
        print(f"  Vehicles:    {num_vehicles}")
        print(f"  Seeds:       {num_seeds}")
        print(f"  Timesteps:   {max_timesteps}")
        print(f"  Obstacles:   {num_obstacles}")
        print(f"  CR:          {cr}")
        print(f"  MARL model:  {model_path or 'N/A'}")
        print(f"  no-OMM model:{no_omm_model or 'N/A'}")
        print(f"  no-GAT model:{no_gat_model or 'N/A'}")
        print("=" * 60)

    start = time.time()

    for config_name, run_func in configs:
        if verbose:
            print(f"\n[{CONFIG_LABELS[config_name].replace(chr(10), ' ')}]")

        for seed in range(num_seeds):
            t0 = time.time()
            result = run_func(base_config, 42 + seed)
            result["seed"] = seed
            all_results.append(result)
            elapsed = time.time() - t0
            if verbose:
                print(f"  Seed {seed}: MSTT={result['final_mstt']:.2f}, "
                      f"wait={result['avg_wait_per_vehicle']:.2f}, "
                      f"trips={result['total_trips']}, "
                      f"time={elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)

    total_time = time.time() - start
    if verbose:
        print(f"\nTotal benchmark time: {total_time:.0f}s")
        print(f"Results saved to: {output_dir}/benchmark_results.csv")

    return df



def plot_comparison_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: str,
    lower_is_better: bool = True,
    show_error_bars: bool = False,
) -> None:
    """Bar chart comparing a metric across all configurations."""
    grouped = df.groupby("config_name")[metric].agg(["mean", "std"]).reset_index()

    # Preserve order
    order = CONFIG_ORDER
    grouped["order"] = grouped["config_name"].map({n: i for i, n in enumerate(order)})
    grouped = grouped.sort_values("order").reset_index(drop=True)

    labels = [CONFIG_LABELS[c] for c in grouped["config_name"]]
    colors = [CONFIG_COLORS[c] for c in grouped["config_name"]]
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(12, 6))
    if show_error_bars:
        bars = ax.bar(
            labels, means, yerr=stds,
            color=colors, edgecolor="white", linewidth=1.5,
            capsize=5, width=0.6,
        )
    else:
        bars = ax.bar(
            labels, means,
            color=colors, edgecolor="white", linewidth=1.5,
            width=0.6,
        )

    # Value labels on bars
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(means) * 0.02),
            f"{val:.2f}",
            ha="center", va="bottom",
            fontweight="bold", fontsize=11,
        )

    # Highlight the best config
    if lower_is_better:
        best_idx = np.argmin(means)
    else:
        best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("#065F46")
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_improvement_percentages(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """Show % improvement of each method over HDV baseline."""
    grouped = df.groupby("config_name").agg({
        "final_mstt": "mean",
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
    }).reset_index()

    # Use HDV as baseline
    baseline = grouped[grouped["config_name"] == "hdv_only"].iloc[0]

    comparisons = []
    for cfg in COMPARISON_ORDER:
        row = grouped[grouped["config_name"] == cfg].iloc[0]
        comparisons.append({
            "config": cfg,
            "mstt_improvement": ((baseline["final_mstt"] - row["final_mstt"]) / baseline["final_mstt"] * 100) if baseline["final_mstt"] > 0 else 0,
            "wait_improvement": ((baseline["avg_wait_per_vehicle"] - row["avg_wait_per_vehicle"]) / baseline["avg_wait_per_vehicle"] * 100) if baseline["avg_wait_per_vehicle"] > 0 else 0,
            "trips_improvement": ((row["total_trips"] - baseline["total_trips"]) / baseline["total_trips"] * 100) if baseline["total_trips"] > 0 else 0,
        })

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparisons))
    width = 0.26

    mstt_bars = ax.bar(
        x - width, [c["mstt_improvement"] for c in comparisons],
        width, label="MSTT reduction", color="#3B82F6",
        edgecolor="white", linewidth=1.5,
    )
    wait_bars = ax.bar(
        x, [c["wait_improvement"] for c in comparisons],
        width, label="Wait time reduction", color="#10B981",
        edgecolor="white", linewidth=1.5,
    )
    trips_bars = ax.bar(
        x + width, [c["trips_improvement"] for c in comparisons],
        width, label="Trip throughput gain", color="#F59E0B",
        edgecolor="white", linewidth=1.5,
    )

    for bars in [mstt_bars, wait_bars, trips_bars]:
        for bar in bars:
            height = bar.get_height()
            offset = 2 if height >= 0 else -4
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:+.1f}%",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, fontweight="bold",
            )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c["config"]] for c in comparisons])
    ax.set_ylabel("Improvement vs HDV baseline (%)", fontsize=11)
    ax.set_title("Performance gains vs HDV-only baseline", fontsize=13, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fontsize=10,
        framealpha=0.95,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_radar_chart(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """Radar chart showing all metrics across configurations."""
    grouped = df.groupby("config_name").agg({
        "final_mstt": "mean",
        "final_mss": "mean",
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
        "mstt_std": "mean",
    }).reset_index()

    # Normalise all metrics to 0-1 (higher=better after normalisation)
    # For metrics where lower is better, we invert
    order = CONFIG_ORDER
    grouped["order"] = grouped["config_name"].map({n: i for i, n in enumerate(order)})
    grouped = grouped.sort_values("order").reset_index(drop=True)

    metrics_def = [
        ("MSTT\n(lower=better)", "final_mstt", True),
        ("Avg Speed\n(higher=better)", "final_mss", False),
        ("Wait Time\n(lower=better)", "avg_wait_per_vehicle", True),
        ("Throughput\n(higher=better)", "total_trips", False),
        ("Reliability\n(lower std=better)", "mstt_std", True),
    ]

    # Compute normalised scores (0 to 1, higher = better)
    scores = {}
    for cfg in order:
        row = grouped[grouped["config_name"] == cfg].iloc[0]
        scores[cfg] = []
        for label, metric, lower_better in metrics_def:
            vals = grouped[metric].values
            if lower_better:
                # Invert: best (lowest) value gets 1.0
                if vals.max() > vals.min():
                    score = (vals.max() - row[metric]) / (vals.max() - vals.min())
                else:
                    score = 1.0
            else:
                if vals.max() > vals.min():
                    score = (row[metric] - vals.min()) / (vals.max() - vals.min())
                else:
                    score = 1.0
            scores[cfg].append(score)

    # Radar plot
    categories = [m[0] for m in metrics_def]
    num_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    for cfg in order:
        values = scores[cfg] + scores[cfg][:1]  # Close polygon
        ax.plot(
            angles, values,
            color=CONFIG_COLORS[cfg], linewidth=2.5,
            label=CONFIG_LABELS[cfg].replace("\n", " "),
        )
        ax.fill(angles, values, color=CONFIG_COLORS[cfg], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Multi-metric Performance Comparison\n(all scores normalised — outer edge is best)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_table(df: pd.DataFrame, save_path: str) -> None:
    """Produce a clean summary table as an image for the report."""
    grouped = df.groupby("config_name").agg({
        "final_mstt": ["mean", "std"],
        "final_mss": "mean",
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
        "mstt_std": "mean",
    }).round(3)

    order = CONFIG_ORDER
    grouped = grouped.reindex(order)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    rows = []
    for cfg in order:
        row_data = grouped.loc[cfg]
        label = CONFIG_LABELS[cfg].replace("\n", " ")
        rows.append([
            label,
            f"{row_data[('final_mstt', 'mean')]:.2f} ± {row_data[('final_mstt', 'std')]:.2f}",
            f"{row_data[('final_mss', 'mean')]:.3f}",
            f"{row_data[('avg_wait_per_vehicle', 'mean')]:.2f}",
            f"{int(row_data[('total_trips', 'mean')])}",
            f"{row_data[('mstt_std', 'mean')]:.3f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Configuration", "MSTT (mean ± std)", "MSS", "Avg Wait", "Trips", "MSTT std"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Header styling
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor("#1E3A8A")
        cell.set_text_props(color="white", weight="bold")

    # Colour-code rows by config
    for row_idx, cfg in enumerate(order, start=1):
        cell = table[(row_idx, 0)]
        cell.set_facecolor(CONFIG_COLORS[cfg])
        cell.set_text_props(color="white", weight="bold")

    metric_specs = [
        (("final_mstt", "mean"), True),           # lower is better
        (("final_mss", "mean"), False),           # higher is better
        (("avg_wait_per_vehicle", "mean"), True), # lower is better
        (("total_trips", "mean"), False),         # higher is better
        (("mstt_std", "mean"), True),             # lower is better
    ]
    for col_idx, (metric_key, lower_is_better) in enumerate(metric_specs, start=1):
        series = grouped[metric_key].astype(float)
        best_cfg = series.idxmin() if lower_is_better else series.idxmax()
        best_row = order.index(best_cfg) + 1
        table[(best_row, col_idx)].set_facecolor("#D1FAE5")

    ax.set_title("Benchmark Summary Table", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")



def plot_safety_collision_rate(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart for collision rate across configurations."""
    plot_comparison_bar(
        df=df,
        metric="collision_rate_per_1000_trips",
        ylabel="Collision Rate (per 1000 trips)",
        title="Collision Rate Comparison (lower is safer)",
        save_path=save_path,
        lower_is_better=True,
    )


def plot_safety_near_miss_rate(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart for near-miss rate across configurations."""
    plot_comparison_bar(
        df=df,
        metric="near_miss_rate_per_1000_trips",
        ylabel="Near-Miss Rate (per 1000 trips)",
        title="Near-Miss Rate Comparison (lower is safer)",
        save_path=save_path,
        lower_is_better=True,
    )


def plot_safety_tradeoff(df: pd.DataFrame, save_path: str) -> None:
    """Scatter plot of efficiency (MSTT) vs collision rate."""
    grouped = df.groupby("config_name").agg({
        "final_mstt": ["mean", "std"],
        "collision_rate_per_1000_trips": ["mean", "std"],
    })

    order = CONFIG_ORDER

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for cfg in order:
        if cfg not in grouped.index:
            continue

        x = grouped.loc[cfg, ("final_mstt", "mean")]
        y = grouped.loc[cfg, ("collision_rate_per_1000_trips", "mean")]
        xerr = grouped.loc[cfg, ("final_mstt", "std")]
        yerr = grouped.loc[cfg, ("collision_rate_per_1000_trips", "std")]

        ax.errorbar(
            x, y,
            xerr=xerr if not np.isnan(xerr) else 0.0,
            yerr=yerr if not np.isnan(yerr) else 0.0,
            fmt="o",
            color=CONFIG_COLORS[cfg],
            ecolor=CONFIG_COLORS[cfg],
            elinewidth=1.5,
            capsize=4,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            alpha=0.95,
            label=CONFIG_LABELS[cfg].replace("\n", " "),
        )

    ax.set_xlabel("MSTT (lower is better)")
    ax.set_ylabel("Collision Rate per 1000 Trips (lower is better)")
    ax.set_title("Safety-Efficiency Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the CAV system")
    parser.add_argument("--vehicles", type=int, default=100, help="Number of vehicles")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--timesteps", type=int, default=500, help="Max timesteps")
    parser.add_argument("--obstacles", type=int, default=NUM_OBSTACLES, help="Number of obstacles")
    parser.add_argument("--cr", type=float, default=COMM_RADIUS, help="Communication radius")
    parser.add_argument("--model", default="results/marl/final_model.pt", help="Path to trained MA2C model")
    parser.add_argument("--model-no-omm", default=None, help="Optional separate model path for Dec-CTDSP + MA2C (no OMM)")
    parser.add_argument("--model-no-gat", default=None, help="Optional separate model path for Dec-CTDSP + MA2C (no GAT)")
    parser.add_argument("--output", default="results/benchmark", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer runs)")
    args = parser.parse_args()

    if args.quick:
        vehicles = 30
        seeds = 2
        timesteps = 200
    else:
        vehicles = args.vehicles
        seeds = args.seeds
        timesteps = args.timesteps

    df = run_benchmark(
        num_vehicles=vehicles,
        num_seeds=seeds,
        max_timesteps=timesteps,
        num_obstacles=args.obstacles,
        cr=args.cr,
        model_path=args.model,
        model_path_no_omm=args.model_no_omm,
        model_path_no_gat=args.model_no_gat,
        output_dir=args.output,
        verbose=True,
    )

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Benchmark Plots")
    print("=" * 60)

    plot_comparison_bar(
        df, "final_mstt",
        "Mean System Travel Time (timesteps)",
        "MSTT Across Configurations",
        os.path.join(args.output, "benchmark_mstt.png"),
        lower_is_better=True,
    )

    plot_comparison_bar(
        df, "avg_wait_per_vehicle",
        "Avg Wait Time per Trip (timesteps)",
        "Wait Time Comparison (proves OMM value)",
        os.path.join(args.output, "benchmark_wait_time.png"),
        lower_is_better=True,
    )

    plot_comparison_bar(
        df, "total_trips",
        "Total Trips Completed",
        "Network Throughput",
        os.path.join(args.output, "benchmark_throughput.png"),
        lower_is_better=False,
        show_error_bars=False,
    )

    plot_comparison_bar(
        df, "mstt_std",
        "MSTT Standard Deviation",
        "Reliability (lower std = more reliable)",
        os.path.join(args.output, "benchmark_reliability.png"),
        lower_is_better=True,
    )

    plot_improvement_percentages(
        df,
        os.path.join(args.output, "benchmark_improvements.png"),
    )

    plot_radar_chart(
        df,
        os.path.join(args.output, "benchmark_radar.png"),
    )

    plot_summary_table(
        df,
        os.path.join(args.output, "benchmark_summary_table.png"),
    )

    plot_safety_collision_rate(
        df,
        os.path.join(args.output, "benchmark_collision_rate.png"),
    )

    plot_safety_near_miss_rate(
        df,
        os.path.join(args.output, "benchmark_near_miss_rate.png"),
    )

    plot_safety_tradeoff(
        df,
        os.path.join(args.output, "benchmark_safety_tradeoff.png"),
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("Final Benchmark Summary")
    print("=" * 60)
    summary = df.groupby("config_name").agg({
        "final_mstt": ["mean", "std"],
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
        "collision_rate_per_1000_trips": "mean",
        "near_miss_rate_per_1000_trips": "mean",
    }).round(3)
    print(summary.to_string())

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
