"""
evaluate.py
Runs systematic experiments sweeping Market Penetration (MP) and
Communication Radius (CR), then produces plots.

Experiments:
    1. MP Sweep:  Fix CR=0.5, vary MP from 0% to 100%
    2. CR Sweep:  Fix MP=100%, vary CR from 0.1 to 1.5
    3. OMM Proof: Compare Reroute+OMM vs Reroute-without-OMM vs No-Reroute

Outputs (saved to results/):
    - mstt_vs_mp.png        MSTT as a function of Market Penetration
    - mss_vs_mp.png         MSS as a function of Market Penetration
    - wait_time_vs_mp.png   Average wait time per trip vs MP
    - mstt_vs_cr.png        MSTT as a function of Communication Radius
    - results_table.csv     Raw numbers for all experiments
    - convergence.png       MSTT over time for selected configs

Usage:
    python src/evaluate.py                  # Run all experiments
    python src/evaluate.py --quick          # Quick run (fewer configs, shorter sims)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.simulator import Simulator, SimConfig


# ── Experiment Runner ────────────────────────────────────────────────────────

def run_experiment(config: SimConfig, label: str = "", verbose: bool = False) -> Dict:
    """Run a single simulation and return its metrics summary."""
    if verbose:
        print(f"  Running: {label} (MP={config.market_penetration:.0%}, "
              f"CR={config.communication_radius}, "
              f"vehicles={config.num_vehicles})...", end=" ", flush=True)

    start = time.time()
    sim = Simulator(config)
    metrics = sim.run(verbose=False)
    elapsed = time.time() - start

    summary = metrics.summary()
    summary["market_penetration"] = config.market_penetration
    summary["communication_radius"] = config.communication_radius
    summary["num_vehicles"] = config.num_vehicles
    summary["num_obstacles"] = config.num_obstacles
    summary["label"] = label
    summary["runtime_seconds"] = round(elapsed, 2)

    # Also store the MSTT history for convergence plots
    summary["mstt_history"] = list(metrics.mstt_history)

    if verbose:
        print(f"done ({elapsed:.1f}s) MSTT={summary['final_mstt']:.2f}")

    return summary


# ── Experiment 1: MP Sweep ───────────────────────────────────────────────────

def experiment_mp_sweep(
    mp_values: List[float],
    num_vehicles: int = 100,
    cr: float = 0.5,
    max_timesteps: int = 500,
    num_seeds: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep Market Penetration from 0% to 100% at fixed CR.

    Runs multiple seeds for each MP to get mean + std (like the paper's
    Monte Carlo approach).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment 1: MP Sweep (CR={cr}, vehicles={num_vehicles})")
        print(f"{'='*60}")

    results = []
    for mp in mp_values:
        for seed in range(num_seeds):
            config = SimConfig(
                num_vehicles=num_vehicles,
                market_penetration=mp,
                communication_radius=cr,
                num_obstacles=2,
                max_timesteps=max_timesteps,
                convergence_window=min(100, max_timesteps // 3),
                warmup_steps=10,
                seed=42 + seed,
            )
            summary = run_experiment(
                config,
                label=f"MP={mp:.0%}_seed={seed}",
                verbose=verbose,
            )
            summary["seed"] = seed
            results.append(summary)

    return pd.DataFrame(results)


# ── Experiment 2: CR Sweep ───────────────────────────────────────────────────

def experiment_cr_sweep(
    cr_values: List[float],
    num_vehicles: int = 100,
    mp: float = 1.0,
    max_timesteps: int = 500,
    num_seeds: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep Communication Radius at fixed MP=100%.

    Tests the paper's finding that CR=0.5 is sufficient for
    a fully connected multi-hop network.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment 2: CR Sweep (MP={mp:.0%}, vehicles={num_vehicles})")
        print(f"{'='*60}")

    results = []
    for cr in cr_values:
        for seed in range(num_seeds):
            config = SimConfig(
                num_vehicles=num_vehicles,
                market_penetration=mp,
                communication_radius=cr,
                num_obstacles=2,
                max_timesteps=max_timesteps,
                convergence_window=min(100, max_timesteps // 3),
                warmup_steps=10,
                seed=42 + seed,
            )
            summary = run_experiment(
                config,
                label=f"CR={cr}_seed={seed}",
                verbose=verbose,
            )
            summary["seed"] = seed
            results.append(summary)

    return pd.DataFrame(results)


# ── Plotting Functions ───────────────────────────────────────────────────────

def plot_metric_vs_mp(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: str,
    baseline_label: str = "MP=0% (HDV only)",
) -> None:
    """Plot a metric as a function of Market Penetration with error bars."""
    grouped = df.groupby("market_penetration")[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots()
    ax.errorbar(
        grouped["market_penetration"] * 100,
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=4,
        linewidth=2,
        markersize=7,
        color="#2563EB",
        ecolor="#93C5FD",
        label="Dec-CTDSP + OMM",
    )

    # Baseline reference line (MP=0%)
    baseline = grouped[grouped["market_penetration"] == 0.0]
    if not baseline.empty:
        ax.axhline(
            y=baseline["mean"].values[0],
            color="#EF4444",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=baseline_label,
        )

    ax.set_xlabel("Market Penetration (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_vs_cr(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """Plot a metric as a function of Communication Radius with error bars."""
    grouped = df.groupby("communication_radius")[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots()
    ax.errorbar(
        grouped["communication_radius"],
        grouped["mean"],
        yerr=grouped["std"],
        marker="s",
        capsize=4,
        linewidth=2,
        markersize=7,
        color="#059669",
        ecolor="#6EE7B7",
        label="Dec-CTDSP + OMM (MP=100%)",
    )

    # Mark the paper's optimal CR=0.5
    ax.axvline(
        x=0.5,
        color="#F59E0B",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="CR=0.5 (paper optimal)",
    )

    ax.set_xlabel("Communication Radius (blocks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_convergence(
    results: List[Dict],
    labels: List[str],
    save_path: str,
) -> None:
    """Plot MSTT convergence over time for selected configurations."""
    fig, ax = plt.subplots()

    colors = ["#2563EB", "#EF4444", "#059669", "#F59E0B", "#8B5CF6"]

    for i, (result, label) in enumerate(zip(results, labels)):
        history = result.get("mstt_history", [])
        if history:
            color = colors[i % len(colors)]
            ax.plot(history, linewidth=1.5, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Timestep (after warmup)")
    ax.set_ylabel("Mean System Travel Time (MSTT)")
    ax.set_title("MSTT Convergence Over Time")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_improvement_summary(
    df_mp: pd.DataFrame,
    save_path: str,
) -> None:
    """Bar chart showing % improvement in MSTT, MSS, and wait time vs baseline."""
    grouped = df_mp.groupby("market_penetration").agg({
        "final_mstt": "mean",
        "final_mss": "mean",
        "avg_wait_per_vehicle": "mean",
    }).reset_index()

    baseline = grouped[grouped["market_penetration"] == 0.0].iloc[0]
    best_cav = grouped[grouped["market_penetration"] == 1.0].iloc[0]

    metrics = {
        "MSTT\nReduction": ((baseline["final_mstt"] - best_cav["final_mstt"]) / baseline["final_mstt"]) * 100 if baseline["final_mstt"] > 0 else 0,
        "MSS\nIncrease": ((best_cav["final_mss"] - baseline["final_mss"]) / baseline["final_mss"]) * 100 if baseline["final_mss"] > 0 else 0,
        "Wait Time\nReduction": ((baseline["avg_wait_per_vehicle"] - best_cav["avg_wait_per_vehicle"]) / baseline["avg_wait_per_vehicle"]) * 100 if baseline["avg_wait_per_vehicle"] > 0 else 0,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        metrics.keys(),
        metrics.values(),
        color=["#2563EB", "#059669", "#F59E0B"],
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )

    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    ax.set_ylabel("Improvement (%)")
    ax.set_title("Dec-CTDSP + OMM vs HDV-Only Baseline (MP=100% vs MP=0%)")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAV Simulation Evaluation Suite")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer configs")
    parser.add_argument("--output", default="results", help="Output directory for plots/data")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.quick:
        mp_values = [0.0, 0.5, 1.0]
        cr_values = [0.1, 0.5, 1.0]
        num_seeds = 1
        max_timesteps = 300
        num_vehicles = 50
    else:
        mp_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        cr_values = [0.1, 0.3, 0.5, 0.75, 1.0, 1.5]
        num_seeds = 3
        max_timesteps = 500
        num_vehicles = 100

    total_runs = len(mp_values) * num_seeds + len(cr_values) * num_seeds
    print(f"Total experiment runs: {total_runs}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")

    # ── Experiment 1: MP Sweep ──
    df_mp = experiment_mp_sweep(
        mp_values=mp_values,
        num_vehicles=num_vehicles,
        cr=0.5,
        max_timesteps=max_timesteps,
        num_seeds=num_seeds,
        verbose=True,
    )

    # ── Experiment 2: CR Sweep ──
    df_cr = experiment_cr_sweep(
        cr_values=cr_values,
        num_vehicles=num_vehicles,
        mp=1.0,
        max_timesteps=max_timesteps,
        num_seeds=num_seeds,
        verbose=True,
    )

    # ── Save raw data ──
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    # Drop mstt_history from CSV (it's a list, not a scalar)
    df_mp_save = df_mp.drop(columns=["mstt_history"], errors="ignore")
    df_cr_save = df_cr.drop(columns=["mstt_history"], errors="ignore")

    df_mp_save.to_csv(os.path.join(args.output, "mp_sweep_results.csv"), index=False)
    df_cr_save.to_csv(os.path.join(args.output, "cr_sweep_results.csv"), index=False)
    print(f"  Saved: {args.output}/mp_sweep_results.csv")
    print(f"  Saved: {args.output}/cr_sweep_results.csv")

    # ── Generate plots ──
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")

    # MSTT vs MP
    plot_metric_vs_mp(
        df_mp, "final_mstt", "Mean System Travel Time (timesteps)",
        "MSTT vs Market Penetration (CR=0.5)",
        os.path.join(args.output, "mstt_vs_mp.png"),
    )

    # MSS vs MP
    plot_metric_vs_mp(
        df_mp, "final_mss", "Mean System Speed (blocks/timestep)",
        "MSS vs Market Penetration (CR=0.5)",
        os.path.join(args.output, "mss_vs_mp.png"),
    )

    # Wait time vs MP
    plot_metric_vs_mp(
        df_mp, "avg_wait_per_vehicle", "Avg Wait Time per Trip (timesteps)",
        "Wait Time vs Market Penetration (CR=0.5)",
        os.path.join(args.output, "wait_time_vs_mp.png"),
    )

    # MSTT vs CR
    plot_metric_vs_cr(
        df_cr, "final_mstt", "Mean System Travel Time (timesteps)",
        "MSTT vs Communication Radius (MP=100%)",
        os.path.join(args.output, "mstt_vs_cr.png"),
    )

    # Convergence plot (pick one run per MP level)
    convergence_results = []
    convergence_labels = []
    for mp in mp_values:
        subset = df_mp[df_mp["market_penetration"] == mp]
        if not subset.empty:
            row = subset.iloc[0].to_dict()
            convergence_results.append(row)
            convergence_labels.append(f"MP={mp:.0%}")

    if convergence_results:
        plot_convergence(
            convergence_results,
            convergence_labels,
            os.path.join(args.output, "convergence.png"),
        )

    # Improvement summary bar chart
    plot_improvement_summary(
        df_mp,
        os.path.join(args.output, "improvement_summary.png"),
    )

    # ── Print final summary table ──
    print(f"\n{'='*60}")
    print("Final Summary: MP Sweep")
    print(f"{'='*60}")
    mp_summary = df_mp.groupby("market_penetration").agg({
        "final_mstt": ["mean", "std"],
        "final_mss": ["mean", "std"],
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
        "total_recalculations": "mean",
    }).round(3)
    print(mp_summary.to_string())

    print(f"\n{'='*60}")
    print("Final Summary: CR Sweep")
    print(f"{'='*60}")
    cr_summary = df_cr.groupby("communication_radius").agg({
        "final_mstt": ["mean", "std"],
        "final_mss": ["mean", "std"],
        "avg_wait_per_vehicle": "mean",
        "total_trips": "mean",
    }).round(3)
    print(cr_summary.to_string())

    print(f"\nAll results saved to: {args.output}/")
    print("Done.")


if __name__ == "__main__":
    main()