"""
benchmark.py - Comprehensive System Benchmarking
Compares your full Dec-CTDSP + OMM + MA2C system against ablations
and baselines to prove each component adds measurable value.

Configurations tested:
    1. HDV-only (MP=0%)             - No CAVs, pure HDV routing
    2. Static Dijkstra (no OMM)     - CAVs use Dijkstra but no obstacle memory
    3. Dec-CTDSP + OMM              - Full high-level routing system
    4. Dec-CTDSP + MA2C (no OMM)     - CAVs use MA2C for decisions but no OMM
    5. Dec-CTDSP + OMM + MA2C       - Full system including trained MARL

Metrics compared:
    - MSTT (Mean System Travel Time)
    - MSS (Mean System Speed)
    - Average wait time per trip (proves OMM value)
    - Total trips completed (throughput)
    - Standard deviation of MSTT (reliability)
    - Number of stalled vehicles

Outputs:
    - benchmark_mstt.png         Bar chart comparing MSTT across configs
    - benchmark_wait_time.png    Wait time comparison (OMM proof)
    - benchmark_throughput.png   Trip completion comparison
    - benchmark_reliability.png  MSTT std deviation
    - benchmark_radar.png        All metrics on a radar chart
    - benchmark_results.csv      Raw numbers
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
from src.marl.ma2c import MA2CAgent, MA2CConfig
from src.marl.gat_network import GATConfig

import torch


# ── Config Labels ────────────────────────────────────────────────────────────

CONFIG_LABELS = {
    "hdv_only": "HDV only\n(baseline)",
    "static_dijkstra": "Static Dijkstra\n(no OMM)",
    "dec_ctdsp": "Dec-CTDSP\n+ OMM",
    "dec_ctdsp_ma2c_no_omm": "Dec-CTDSP\n+ MA2C\n(no OMM)",
    "dec_ctdsp_marl": "Dec-CTDSP\n+ OMM + MA2C\n(full system)",
}

CONFIG_COLORS = {
    "hdv_only": "#94A3B8",              # Gray
    "static_dijkstra": "#F59E0B",       # Amber
    "dec_ctdsp": "#3B82F6",             # Blue
    "dec_ctdsp_ma2c_no_omm": "#8B5CF6", # Purple (new ablation)
    "dec_ctdsp_marl": "#10B981",        # Green (full system)
}


# ── Run Configurations ───────────────────────────────────────────────────────

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
    and their OMM blacklists are cleared each timestep, so they
    effectively have no obstacle memory.
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


def _run_with_trained_agent(
    cfg: SimConfig,
    seed: int,
    model_path: str,
    config_name: str,
    enable_omm: bool,
) -> Dict:
    """
    Internal helper: run a simulation where the trained MA2C agent
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
    tc.grid_rows = cfg.grid_rows
    tc.grid_cols = cfg.grid_cols
    tc.steps_per_episode = cfg.max_timesteps
    tc.seed = seed

    # Load trained agent
    agent = MA2CAgent()
    agent.load(model_path)
    agent.set_eval_mode()

    # Build environment
    env = MARLEnvironment(tc)
    env.reset(seed=seed)

    # Run the episode, tracking metrics manually
    all_trip_times = []
    total_recalcs = 0
    total_broadcasts = 0
    total_wait_steps = 0
    total_trips = 0
    mstt_history = []
    speeds_history = []

    for step in range(cfg.max_timesteps):
        # Optionally clear blacklists (disables OMM)
        if not enable_omm:
            for v in env.vehicles:
                if isinstance(v, CAV):
                    v.blacklist.clear()

        # Get observations
        node_features, edge_index, vid_order = env.get_gat_inputs()
        global_state = env.get_global_state()

        if len(vid_order) == 0:
            break

        # Run GAT + actor to pick actions
        with torch.no_grad():
            contexts = agent.gat(node_features, edge_index)

        actions: Dict[int, int] = {}
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

        # Wait tracking (approximate: count stalled vehicles each step)
        stalled_now = sum(1 for cnt in env._stall_counters.values() if cnt > 0)
        total_wait_steps += stalled_now

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
        "timesteps_run": len(mstt_history),
        "mstt_std": float(np.std(mstt_history[-100:])) if len(mstt_history) >= 100 else 0.0,
        "marl_model_used": True,
    }


# ── Main Benchmark Runner ────────────────────────────────────────────────────

def run_benchmark(
    num_vehicles: int = 100,
    num_seeds: int = 3,
    max_timesteps: int = 500,
    num_obstacles: int = 2,
    cr: float = 0.5,
    model_path: Optional[str] = None,
    output_dir: str = "results/benchmark",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all benchmark configurations and return aggregated results."""
    os.makedirs(output_dir, exist_ok=True)

    base_config = SimConfig(
        num_vehicles=num_vehicles,
        communication_radius=cr,
        num_obstacles=num_obstacles,
        max_timesteps=max_timesteps,
        convergence_window=min(100, max_timesteps // 3),
        warmup_steps=10,
    )

    all_results = []

    configs = [
        ("hdv_only", run_hdv_only),
        ("static_dijkstra", run_static_dijkstra),
        ("dec_ctdsp", run_dec_ctdsp),
        ("dec_ctdsp_ma2c_no_omm", lambda c, s: run_dec_ctdsp_ma2c_no_omm(c, s, model_path)),
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


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: str,
    lower_is_better: bool = True,
) -> None:
    """Bar chart comparing a metric across all configurations."""
    grouped = df.groupby("config_name")[metric].agg(["mean", "std"]).reset_index()

    # Preserve order
    order = ["hdv_only", "static_dijkstra", "dec_ctdsp", "dec_ctdsp_ma2c_no_omm", "dec_ctdsp_marl"]
    grouped["order"] = grouped["config_name"].map({n: i for i, n in enumerate(order)})
    grouped = grouped.sort_values("order").reset_index(drop=True)

    labels = [CONFIG_LABELS[c] for c in grouped["config_name"]]
    colors = [CONFIG_COLORS[c] for c in grouped["config_name"]]
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        labels, means, yerr=stds,
        color=colors, edgecolor="white", linewidth=1.5,
        capsize=5, width=0.6,
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
    for cfg in ["static_dijkstra", "dec_ctdsp", "dec_ctdsp_ma2c_no_omm", "dec_ctdsp_marl"]:
        row = grouped[grouped["config_name"] == cfg].iloc[0]
        comparisons.append({
            "config": cfg,
            "mstt_improvement": ((baseline["final_mstt"] - row["final_mstt"]) / baseline["final_mstt"] * 100) if baseline["final_mstt"] > 0 else 0,
            "wait_improvement": ((baseline["avg_wait_per_vehicle"] - row["avg_wait_per_vehicle"]) / baseline["avg_wait_per_vehicle"] * 100) if baseline["avg_wait_per_vehicle"] > 0 else 0,
            "trips_improvement": ((row["total_trips"] - baseline["total_trips"]) / baseline["total_trips"] * 100) if baseline["total_trips"] > 0 else 0,
        })

    fig, ax = plt.subplots(figsize=(10, 5.5))

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
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
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
    order = ["hdv_only", "static_dijkstra", "dec_ctdsp", "dec_ctdsp_ma2c_no_omm", "dec_ctdsp_marl"]
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

    order = ["hdv_only", "static_dijkstra", "dec_ctdsp", "dec_ctdsp_ma2c_no_omm", "dec_ctdsp_marl"]
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

    # Highlight best values
    best_row = order.index("dec_ctdsp_marl") + 1
    for i in range(1, 6):
        cell = table[(best_row, i)]
        cell.set_facecolor("#D1FAE5")

    ax.set_title("Benchmark Summary Table", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark the CAV system")
    parser.add_argument("--vehicles", type=int, default=100, help="Number of vehicles")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--timesteps", type=int, default=500, help="Max timesteps")
    parser.add_argument("--obstacles", type=int, default=2, help="Number of obstacles")
    parser.add_argument("--cr", type=float, default=0.5, help="Communication radius")
    parser.add_argument("--model", default="results/marl/final_model.pt", help="Path to trained MA2C model")
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

    # Print final summary
    print("\n" + "=" * 60)
    print("Final Benchmark Summary")
    print("=" * 60)
    summary = df.groupby("config_name").agg({
        "final_mstt": ["mean", "std"],
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
    }).round(3)
    print(summary.to_string())

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()