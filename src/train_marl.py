"""
train_marl.py - Training Script for MA2C with GAT
Connects the MARL components to the simulator and runs the
Centralized Training with Decentralized Execution (CTDE) loop.

Training loop:
    1. Reset the simulator for a new episode
    2. For each timestep in the episode:
       a. Build observations for each CAV
       b. Run GAT to get context vectors
       c. Actor selects actions per CAV
       d. Execute actions in the simulator
       e. Compute rewards
       f. Critic evaluates (with global state)
       g. Store experience in rollout buffer
    3. After rollout_length steps, compute advantages and update networks
    4. Log training metrics

Actions (what the MA2C agent controls):
    0: Follow Dec-CTDSP route — continue on the planned route
    1: Alternative route — take second-best path via modified Dijkstra
    2: Wait — stay at current intersection for one timestep
    3: Reroute — trigger a fresh Dec-CTDSP computation immediately

Usage:
    python src/train_marl.py                    # Full training
    python src/train_marl.py --episodes 20      # Short training run
    python src/train_marl.py --eval             # Evaluate saved model
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_network import GridNetwork, NetworkConfig, speed_density
from src.environment.vehicle import (
    CAV, HDV, Vehicle, VehicleFactory, VehicleState, VehicleType,
)
from src.communication.comm_manager import CommunicationManager, CommConfig
from src.routing.dec_ctdsp import dec_ctdsp_route
from src.environment.simulator import ObstacleManager, SimConfig
from src.marl.reward import RewardCalculator, RewardConfig, shortest_path_distance
from src.marl.gat_network import (
    GATConfig, GATNetwork, ObservationBuilder, build_cluster_graph,
)
from src.marl.ma2c import MA2CAgent, MA2CConfig, GlobalStateBuilder


# ── Training Configuration ───────────────────────────────────────────────────

class TrainConfig:
    """All training hyperparameters."""
    # Environment
    num_vehicles: int = 30
    market_penetration: float = 1.0
    communication_radius: float = 0.5
    num_obstacles: int = 2
    grid_rows: int = 6
    grid_cols: int = 6

    # Training
    num_episodes: int = 200
    steps_per_episode: int = 100
    rollout_length: int = 16
    learning_rate: float = 3e-4
    gamma: float = 0.99
    seed: int = 42

    # Logging
    log_interval: int = 10
    save_interval: int = 50
    save_dir: str = "results/marl"

    # Eval
    eval_episodes: int = 5
    eval_interval: int = 25


# ── Training Environment ─────────────────────────────────────────────────────

class MARLEnvironment:
    """
    Wraps the grid simulation for MARL training.

    Unlike the original Simulator which runs autonomously, this
    environment exposes a step() interface where the MA2C agent
    controls CAV decisions at each intersection.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.timestep = 0

        # Build network
        net_config = NetworkConfig(
            rows=config.grid_rows,
            cols=config.grid_cols,
        )
        self.network = GridNetwork(net_config)

        # Communication
        self.comm = CommunicationManager(
            CommConfig(communication_radius=config.communication_radius)
        )

        # Obstacles
        self.obstacle_mgr = ObstacleManager(
            network=self.network,
            num_obstacles=config.num_obstacles,
            seed=config.seed,
        )

        # Reward calculator
        self.reward_calc = RewardCalculator()

        # Vehicles
        self.vehicles: List[Vehicle] = []
        self.cavs: List[CAV] = []

        # Per-vehicle tracking for rewards
        self._prev_distances: Dict[int, float] = {}
        self._stall_counters: Dict[int, int] = {}
        self._trip_starts: Dict[int, int] = {}

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment for a new episode."""
        actual_seed = seed if seed is not None else self.config.seed
        np.random.seed(actual_seed)

        # Rebuild network (clears all vehicle placements and obstacles)
        net_config = NetworkConfig(
            rows=self.config.grid_rows,
            cols=self.config.grid_cols,
        )
        self.network = GridNetwork(net_config)
        self.obstacle_mgr = ObstacleManager(
            network=self.network,
            num_obstacles=self.config.num_obstacles,
            seed=actual_seed + 1,
        )

        # Create fleet
        self.vehicles = VehicleFactory.create_fleet(
            num_vehicles=self.config.num_vehicles,
            market_penetration=self.config.market_penetration,
            origins=self.network.origins,
            destinations=self.network.destinations,
            seed=actual_seed,
        )

        # Extract CAVs and set up routing
        self.cavs = [v for v in self.vehicles if isinstance(v, CAV)]
        for cav in self.cavs:
            cav.set_routing_function(dec_ctdsp_route)
            cav.compute_route(self.network, timestep=0, cluster_vehicles=[])
            cav.depart(timestep=0)
            cav.speed = self.network.config.speed_limit

        # HDVs get their naive routes
        for v in self.vehicles:
            if isinstance(v, HDV):
                v.compute_route(self.network, timestep=0)
                v.depart(timestep=0)
                v.speed = self.network.config.speed_limit

        # Init tracking
        self._prev_distances = {}
        self._stall_counters = {}
        self._trip_starts = {}
        for cav in self.cavs:
            dist = shortest_path_distance(
                self.network, cav.current_node, cav.destination
            )
            self._prev_distances[cav.vehicle_id] = dist
            self._stall_counters[cav.vehicle_id] = 0
            self._trip_starts[cav.vehicle_id] = 0

        self.timestep = 0

    def step(self, actions: Dict[int, int]) -> Dict[int, Tuple[float, bool, dict]]:
        """
        Execute one timestep with MA2C-selected actions for each CAV.

        Args:
            actions: {vehicle_id: action_id} for each CAV.
                     0=follow, 1=alternative, 2=wait, 3=reroute

        Returns:
            {vehicle_id: (reward, done, info)} for each CAV.
        """
        self.timestep += 1

        # Update environment
        self.obstacle_mgr.update(self.timestep)
        self.comm.update_clusters(self.vehicles, self.network)
        self.comm.propagate_confirmations(self.vehicles, self.network, self.timestep)
        self.comm.decay_all_blacklists(self.vehicles, self.timestep)

        # Move HDVs (they don't use MARL actions)
        for v in self.vehicles:
            if isinstance(v, HDV) and v.state == VehicleState.EN_ROUTE:
                self._move_hdv(v)

        # Execute CAV actions and collect rewards
        results: Dict[int, Tuple[float, bool, dict]] = {}

        for cav in self.cavs:
            if cav.state != VehicleState.EN_ROUTE:
                results[cav.vehicle_id] = (0.0, True, {"status": "not_active"})
                continue

            action = actions.get(cav.vehicle_id, 0)
            reward, done, info = self._execute_cav_action(cav, action)
            results[cav.vehicle_id] = (reward, done, info)

        # Handle arrivals for all vehicles
        self._handle_arrivals()

        return results

    def _execute_cav_action(
        self, cav: CAV, action: int
    ) -> Tuple[float, bool, dict]:
        """Execute a single action for one CAV and compute reward."""
        prev_dist = self._prev_distances.get(cav.vehicle_id, 0.0)
        is_stalled = False

        # Check for obstacles
        next_node = cav.get_next_node()
        blocked_ahead = (
            next_node is not None
            and self.network.is_node_blocked(next_node)
        )

        if blocked_ahead and isinstance(cav, CAV):
            self.comm.broadcast_obstacle(
                cav, next_node, self.vehicles, self.timestep
            )

        # Execute action
        if action == 0:  # Follow Dec-CTDSP route
            if blocked_ahead:
                is_stalled = True
                cluster = self.comm.get_cluster_members(
                    cav.vehicle_id, self.vehicles
                )
                cav.compute_route(
                    self.network, self.timestep, cluster_vehicles=cluster
                )
            else:
                self._advance_vehicle(cav)

        elif action == 1:  # Alternative route
            cav._fallback_dijkstra(self.network, cav.get_blacklisted_nodes())
            cluster = self.comm.get_cluster_members(
                cav.vehicle_id, self.vehicles
            )
            cav.compute_route(
                self.network, self.timestep, cluster_vehicles=cluster
            )
            if not blocked_ahead:
                self._advance_vehicle(cav)
            else:
                is_stalled = True

        elif action == 2:  # Wait
            is_stalled = True

        elif action == 3:  # Reroute
            cluster = self.comm.get_cluster_members(
                cav.vehicle_id, self.vehicles
            )
            cav.compute_route(
                self.network, self.timestep, cluster_vehicles=cluster
            )
            if not blocked_ahead:
                self._advance_vehicle(cav)
            else:
                is_stalled = True

        # Update stall counter
        if is_stalled:
            self._stall_counters[cav.vehicle_id] = (
                self._stall_counters.get(cav.vehicle_id, 0) + 1
            )
        else:
            self._stall_counters[cav.vehicle_id] = 0

        # Compute current distance
        curr_dist = shortest_path_distance(
            self.network, cav.current_node, cav.destination
        )

        # Compute link density ratio
        next_n = cav.get_next_node()
        if next_n and self.network.graph.has_edge(cav.current_node, next_n):
            density = self.network.get_edge_density(cav.current_node, next_n)
            capacity = self.network.graph.edges[cav.current_node, next_n]["capacity"]
            density_ratio = density / max(1, capacity)
        else:
            density_ratio = 0.0

        # Check if arrived
        just_arrived = cav.has_reached_destination()
        trip_duration = self.timestep - self._trip_starts.get(cav.vehicle_id, 0)

        # Compute reward
        reward, breakdown = self.reward_calc.compute(
            vehicle=cav,
            network=self.network,
            prev_distance=prev_dist,
            curr_distance=curr_dist,
            is_stalled=is_stalled,
            consecutive_stall_steps=self._stall_counters.get(cav.vehicle_id, 0),
            just_arrived=just_arrived,
            trip_duration=float(trip_duration),
            link_density_ratio=density_ratio,
        )

        # Update tracking
        self._prev_distances[cav.vehicle_id] = curr_dist

        info = {
            "action": action,
            "prev_dist": prev_dist,
            "curr_dist": curr_dist,
            "is_stalled": is_stalled,
            "just_arrived": just_arrived,
            "breakdown": breakdown,
        }

        return reward, just_arrived, info

    def _advance_vehicle(self, v: Vehicle) -> None:
        """Move a vehicle one step forward on its route."""
        next_node = v.get_next_node()
        if next_node is None:
            return

        prev = v.current_node
        if self.network.graph.has_edge(prev, next_node):
            self.network.remove_vehicle_from_link(v.vehicle_id, prev, next_node)
            density = self.network.get_edge_density(prev, next_node)
            v.speed = speed_density(
                density,
                self.network.config.link_capacity,
                self.network.config.speed_limit,
            )

        v.advance_route()
        v.total_travel_time += 1

        next_next = v.get_next_node()
        if next_next and self.network.graph.has_edge(v.current_node, next_next):
            self.network.place_vehicle_on_link(
                v.vehicle_id, v.current_node, next_next
            )

    def _move_hdv(self, v: HDV) -> None:
        """Move an HDV one step (simple, no MARL)."""
        if v.state != VehicleState.EN_ROUTE:
            return
        next_node = v.get_next_node()
        if next_node is None:
            return
        if self.network.is_node_blocked(next_node):
            return
        self._advance_vehicle(v)

    def _handle_arrivals(self) -> None:
        """Reset arrived vehicles for new trips."""
        for v in self.vehicles:
            if v.state != VehicleState.EN_ROUTE:
                continue
            if v.has_reached_destination():
                v.arrive(self.timestep)
                VehicleFactory.reassign_destination(v, self.network.destinations)
                if isinstance(v, CAV):
                    v.set_routing_function(dec_ctdsp_route)
                    v.compute_route(self.network, self.timestep, cluster_vehicles=[])
                elif isinstance(v, HDV):
                    v.compute_route(self.network, self.timestep)
                v.depart(self.timestep)
                v.speed = self.network.config.speed_limit
                # Reset tracking
                dist = shortest_path_distance(
                    self.network, v.current_node, v.destination
                )
                self._prev_distances[v.vehicle_id] = dist
                self._stall_counters[v.vehicle_id] = 0
                self._trip_starts[v.vehicle_id] = self.timestep

    # ── Observation Building ─────────────────────────────────────────────

    def get_observations(self) -> Dict[int, torch.Tensor]:
        """Build local observation tensors for all active CAVs."""
        obs = {}
        for cav in self.cavs:
            if cav.state != VehicleState.EN_ROUTE:
                continue

            dist = shortest_path_distance(
                self.network, cav.current_node, cav.destination
            )
            trip_time = self.timestep - self._trip_starts.get(cav.vehicle_id, 0)

            local = ObservationBuilder.build_local_obs(
                current_node=cav.current_node,
                destination=cav.destination,
                speed=cav.speed,
                dist_remaining=dist,
                time_elapsed=float(trip_time),
                num_blacklisted=len(cav.get_blacklisted_nodes()),
                grid_rows=self.config.grid_rows,
                grid_cols=self.config.grid_cols,
            )
            obs[cav.vehicle_id] = torch.tensor(local, dtype=torch.float32)
        return obs

    def get_gat_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Build GAT input: stacked node features + edge index.

        Returns:
            (node_features, edge_index, vehicle_id_order)
        """
        active_cavs = [
            c for c in self.cavs if c.state == VehicleState.EN_ROUTE
        ]
        if not active_cavs:
            return torch.zeros((0, 8)), torch.zeros((2, 0), dtype=torch.long), []

        features = []
        vid_order = []

        for cav in active_cavs:
            dist = shortest_path_distance(
                self.network, cav.current_node, cav.destination
            )
            trip_time = self.timestep - self._trip_starts.get(cav.vehicle_id, 0)

            local = ObservationBuilder.build_local_obs(
                current_node=cav.current_node,
                destination=cav.destination,
                speed=cav.speed,
                dist_remaining=dist,
                time_elapsed=float(trip_time),
                num_blacklisted=len(cav.get_blacklisted_nodes()),
                grid_rows=self.config.grid_rows,
                grid_cols=self.config.grid_cols,
            )
            features.append(local)
            vid_order.append(cav.vehicle_id)

        node_features = torch.tensor(features, dtype=torch.float32)
        edge_index = build_cluster_graph(list(range(len(active_cavs))))

        return node_features, edge_index, vid_order

    def get_global_state(self) -> torch.Tensor:
        """Build the global state vector for the critic."""
        active = [v for v in self.vehicles if v.state == VehicleState.EN_ROUTE]
        speeds = [v.speed for v in active if v.speed > 0]

        densities = self.network.get_link_densities()
        density_vals = list(densities.values())
        avg_density = np.mean(density_vals) / max(1, self.network.config.link_capacity) if density_vals else 0
        max_density = max(density_vals) / max(1, self.network.config.link_capacity) if density_vals else 0

        num_cavs = sum(1 for v in self.vehicles if v.vehicle_type == VehicleType.CAV)
        stalled = sum(1 for vid, cnt in self._stall_counters.items() if cnt > 0)

        gs = GlobalStateBuilder.build(
            total_vehicles=len(active),
            avg_link_density=float(avg_density),
            max_link_density=float(max_density),
            num_obstacles=len(self.network.get_blocked_nodes()),
            avg_system_speed=float(np.mean(speeds)) if speeds else 0.0,
            cav_fraction=num_cavs / max(1, len(self.vehicles)),
            num_clusters=self.comm.get_num_clusters(),
            avg_cluster_size=float(np.mean(self.comm.get_cluster_sizes())) if self.comm.clusters else 0,
            total_trips_completed=sum(v.trips_completed for v in self.vehicles),
            avg_mstt=0.0,
            num_stalled=stalled,
            total_recalculations=sum(
                v.num_route_recalculations for v in self.vehicles
                if isinstance(v, CAV)
            ),
        )
        return torch.tensor(gs, dtype=torch.float32)


# ── Training Loop ────────────────────────────────────────────────────────────

def train(config: TrainConfig, verbose: bool = True) -> MA2CAgent:
    """
    Run the full MARL training loop.

    Returns the trained MA2C agent.
    """
    os.makedirs(config.save_dir, exist_ok=True)

    # Initialise
    ma2c_config = MA2CConfig(learning_rate=config.learning_rate, gamma=config.gamma)
    agent = MA2CAgent(ma2c_config)
    env = MARLEnvironment(config)

    # Training metrics
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    loss_history: List[Dict] = []

    if verbose:
        print("=" * 60)
        print("MA2C Training")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  Steps/episode: {config.steps_per_episode}")
        print(f"  Vehicles: {config.num_vehicles} (MP={config.market_penetration:.0%})")
        print(f"  Rollout length: {config.rollout_length}")
        print("=" * 60)

    start_time = time.time()

    for episode in range(config.num_episodes):
        env.reset(seed=config.seed + episode)
        agent.set_train_mode()

        ep_reward = 0.0
        ep_steps = 0
        update_count = 0

        for step in range(config.steps_per_episode):
            # Get observations
            node_features, edge_index, vid_order = env.get_gat_inputs()
            global_state = env.get_global_state()

            if len(vid_order) == 0:
                break

            # GAT forward pass
            with torch.no_grad():
                contexts = agent.gat(node_features, edge_index)

            # Select actions for each CAV
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
                action, log_prob, value = agent.act(
                    local_obs, context, global_state
                )
                actions[vid] = action

                # Store in rollout (we store per first active CAV for simplicity,
                # or could maintain per-agent rollouts)
                agent.rollout.add(
                    local_obs, context, global_state,
                    torch.tensor(action), log_prob, 0.0, value, False
                )

            # Step environment
            results = env.step(actions)

            # Update rewards in rollout
            total_step_reward = 0.0
            for vid, (reward, done, info) in results.items():
                total_step_reward += reward

            # Set reward for the last batch of rollout entries
            num_entries = len(vid_order)
            for i in range(num_entries):
                idx = len(agent.rollout) - num_entries + i
                if 0 <= idx < len(agent.rollout.rewards):
                    vid = vid_order[i]
                    r = results.get(vid, (0.0, False, {}))[0]
                    agent.rollout.rewards[idx] = r
                    agent.rollout.dones[idx] = results.get(vid, (0, False, {}))[1]

            ep_reward += total_step_reward
            ep_steps += 1

            # Update networks when rollout is full
            if len(agent.rollout) >= config.rollout_length:
                losses = agent.update()
                loss_history.append(losses)
                update_count += 1

        # End of episode
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        # Clear any leftover rollout data
        if len(agent.rollout) > 0:
            losses = agent.update()
            loss_history.append(losses)

        # Logging
        if verbose and (episode + 1) % config.log_interval == 0:
            recent_rewards = episode_rewards[-config.log_interval:]
            avg_reward = np.mean(recent_rewards)
            recent_loss = np.mean([l["total_loss"] for l in loss_history[-10:]]) if loss_history else 0
            elapsed = time.time() - start_time
            print(
                f"  Episode {episode+1:4d}/{config.num_episodes} | "
                f"avg_reward={avg_reward:8.1f} | "
                f"loss={recent_loss:.4f} | "
                f"updates={agent.total_updates} | "
                f"time={elapsed:.0f}s"
            )

        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            path = os.path.join(config.save_dir, f"checkpoint_ep{episode+1}.pt")
            agent.save(path)
            if verbose:
                print(f"    Saved checkpoint: {path}")

        # Periodic evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_reward = evaluate(agent, config, num_episodes=config.eval_episodes)
            if verbose:
                print(f"    Eval avg reward: {eval_reward:.1f}")

    # Save final model
    final_path = os.path.join(config.save_dir, "final_model.pt")
    agent.save(final_path)
    if verbose:
        print(f"\nTraining complete. Model saved to {final_path}")
        print(f"Total updates: {agent.total_updates}")
        print(f"Total time: {time.time() - start_time:.0f}s")

    # Save training curves
    _save_training_curves(episode_rewards, loss_history, config.save_dir)

    return agent


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    agent: MA2CAgent,
    config: TrainConfig,
    num_episodes: int = 5,
) -> float:
    """Run evaluation episodes with deterministic actions."""
    agent.set_eval_mode()
    env = MARLEnvironment(config)

    total_rewards = []

    for ep in range(num_episodes):
        env.reset(seed=1000 + ep)
        ep_reward = 0.0

        for step in range(config.steps_per_episode):
            node_features, edge_index, vid_order = env.get_gat_inputs()
            global_state = env.get_global_state()

            if len(vid_order) == 0:
                break

            with torch.no_grad():
                contexts = agent.gat(node_features, edge_index)

            actions = {}
            local_obs_map = env.get_observations()
            vid_to_idx = {vid: i for i, vid in enumerate(vid_order)}

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

            results = env.step(actions)
            ep_reward += sum(r for r, _, _ in results.values())

        total_rewards.append(ep_reward)

    agent.set_train_mode()
    return float(np.mean(total_rewards))


# ── Training Curves ──────────────────────────────────────────────────────────

def _save_training_curves(
    rewards: List[float],
    losses: List[Dict],
    save_dir: str,
) -> None:
    """Save training reward and loss plots."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve
    ax1.plot(rewards, alpha=0.3, color="#2563EB", linewidth=0.8)
    # Smoothed
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(rewards)), smoothed, color="#2563EB", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Reward")
    ax1.grid(True, alpha=0.3)

    # Loss curve
    if losses:
        total_losses = [l["total_loss"] for l in losses]
        ax2.plot(total_losses, alpha=0.3, color="#EF4444", linewidth=0.8)
        if len(total_losses) > 10:
            window = min(20, len(total_losses) // 5)
            smoothed_loss = np.convolve(
                total_losses, np.ones(window) / window, mode="valid"
            )
            ax2.plot(
                range(window - 1, len(total_losses)),
                smoothed_loss, color="#EF4444", linewidth=2,
            )
        ax2.set_xlabel("Update Step")
        ax2.set_ylabel("Total Loss")
        ax2.set_title("Training Loss")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MA2C agent")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=60, help="Steps per episode")
    parser.add_argument("--vehicles", type=int, default=20, help="Number of vehicles")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model")
    parser.add_argument("--save-dir", default="results/marl", help="Save directory")
    args = parser.parse_args()

    config = TrainConfig()
    config.num_episodes = args.episodes
    config.steps_per_episode = args.steps
    config.num_vehicles = args.vehicles
    config.save_dir = args.save_dir
    config.log_interval = max(1, args.episodes // 10)
    config.save_interval = max(1, args.episodes // 4)
    config.eval_interval = max(1, args.episodes // 4)

    if args.eval:
        # Load and evaluate
        model_path = os.path.join(args.save_dir, "final_model.pt")
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}. Train first.")
            sys.exit(1)
        agent = MA2CAgent()
        agent.load(model_path)
        avg_reward = evaluate(agent, config, num_episodes=10)
        print(f"Evaluation avg reward: {avg_reward:.1f}")
    else:
        # Train
        agent = train(config, verbose=True)