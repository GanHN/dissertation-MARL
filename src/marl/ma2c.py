"""
ma2c.py - Multi-Agent Advantage Actor-Critic (MA2C)
The MARL algorithm that learns cooperative driving policies using
Centralized Training with Decentralized Execution (CTDE).

Architecture:
    Actor:  GAT context + local obs -> action probabilities
            (decentralized — runs on each CAV independently)

    Critic: GAT context + local obs + global state -> value estimate
            (centralized — uses full system info during training only)

Action space (discrete, 4 actions):
    0: Follow Dec-CTDSP route (default)
    1: Take alternative route (second-best path)
    2: Wait one timestep at current intersection
    3: Reroute (trigger fresh Dec-CTDSP computation)

Training uses CTDE:
    - During training: critic sees global traffic state
    - During execution: actor uses only local obs + GNN context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.marl.gat_network import GATConfig, GATNetwork, build_cluster_graph


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class MA2CConfig:
    """Configuration for the MA2C algorithm."""
    # Dimensions
    local_obs_dim: int = 8       # Must match GATConfig
    context_dim: int = 32        # Must match GATConfig
    global_state_dim: int = 16   # Extra global info for critic
    num_actions: int = 4         # Discrete action space size

    # Network architecture
    actor_hidden: int = 64
    critic_hidden: int = 128

    # Training hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda for advantage estimation
    entropy_coeff: float = 0.01  # Entropy bonus for exploration
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5   # Gradient clipping

    # Rollout
    rollout_length: int = 32     # Steps before each update

    # GAT
    gat_config: GATConfig = field(default_factory=GATConfig)


# ── Actor Network ────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Decentralized policy network.

    Input:  local observation + GAT context vector
    Output: action probabilities over discrete action space

    During execution, this runs independently on each CAV
    using only local information and neighbour data (via GNN).
    """

    def __init__(self, config: MA2CConfig):
        super().__init__()
        input_dim = config.local_obs_dim + config.context_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.actor_hidden),
            nn.ReLU(),
            nn.LayerNorm(config.actor_hidden),
            nn.Linear(config.actor_hidden, config.actor_hidden),
            nn.ReLU(),
            nn.Linear(config.actor_hidden, config.num_actions),
        )

    def forward(self, local_obs: torch.Tensor, context: torch.Tensor) -> Categorical:
        """
        Compute action distribution.

        Args:
            local_obs: [batch, local_obs_dim] local observation.
            context:   [batch, context_dim] GAT context vector.

        Returns:
            Categorical distribution over actions.
        """
        x = torch.cat([local_obs, context], dim=-1)
        logits = self.network(x)
        return Categorical(logits=logits)


# ── Critic Network ───────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Centralized value function (used during training only).

    Input:  local observation + GAT context + global state info
    Output: scalar value estimate V(s)

    The global state info gives the critic access to system-wide
    information (total vehicles, average density, etc.) that
    individual agents can't observe. This is the "centralized"
    part of CTDE.
    """

    def __init__(self, config: MA2CConfig):
        super().__init__()
        input_dim = config.local_obs_dim + config.context_dim + config.global_state_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.critic_hidden),
            nn.ReLU(),
            nn.LayerNorm(config.critic_hidden),
            nn.Linear(config.critic_hidden, config.critic_hidden),
            nn.ReLU(),
            nn.Linear(config.critic_hidden, 1),
        )

    def forward(
        self,
        local_obs: torch.Tensor,
        context: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value estimate.

        Args:
            local_obs:    [batch, local_obs_dim]
            context:      [batch, context_dim]
            global_state: [batch, global_state_dim]

        Returns:
            values: [batch, 1] value estimates.
        """
        x = torch.cat([local_obs, context, global_state], dim=-1)
        return self.network(x)


# ── Global State Builder ────────────────────────────────────────────────────

class GlobalStateBuilder:
    """
    Builds the global state vector used by the critic during training.

    This includes system-wide information that no single agent can see:
        - Total vehicles on network
        - Average link density across all links
        - Number of active obstacles
        - Average system speed
        - Fraction of vehicles that are CAVs
        - Number of communication clusters
        - Average cluster size
        - Various normalised traffic statistics
    """

    @staticmethod
    def build(
        total_vehicles: int,
        avg_link_density: float,
        max_link_density: float,
        num_obstacles: int,
        avg_system_speed: float,
        cav_fraction: float,
        num_clusters: int,
        avg_cluster_size: float,
        total_trips_completed: int,
        avg_mstt: float,
        num_stalled: int,
        total_recalculations: int,
        max_vehicles: int = 100,
        max_obstacles: int = 10,
        max_clusters: int = 20,
        max_mstt: float = 50.0,
    ) -> List[float]:
        """Build normalised global state vector (16 features)."""
        return [
            total_vehicles / max(1, max_vehicles),
            avg_link_density,
            max_link_density,
            num_obstacles / max(1, max_obstacles),
            avg_system_speed,
            cav_fraction,
            num_clusters / max(1, max_clusters),
            avg_cluster_size / max(1, max_vehicles),
            min(total_trips_completed / 1000.0, 1.0),
            min(avg_mstt / max_mstt, 1.0),
            num_stalled / max(1, max_vehicles),
            min(total_recalculations / 500.0, 1.0),
            # Padding to reach 16 dimensions (reserved for future use)
            0.0, 0.0, 0.0, 0.0,
        ]


# ── Rollout Storage ──────────────────────────────────────────────────────────

class RolloutStorage:
    """
    Stores experience tuples for a batch of agents over a rollout.

    Each entry is one (state, action, reward, value, log_prob) tuple
    for one agent at one timestep. After rollout_length steps, the
    data is used to compute advantages and update the networks.
    """

    def __init__(self):
        self.local_obs: List[torch.Tensor] = []
        self.contexts: List[torch.Tensor] = []
        self.global_states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(
        self,
        local_obs: torch.Tensor,
        context: torch.Tensor,
        global_state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Add one experience tuple."""
        self.local_obs.append(local_obs.detach())
        self.contexts.append(context.detach())
        self.global_states.append(global_state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and discounted returns.

        Uses Generalized Advantage Estimation (GAE) for lower-variance
        advantage estimates, which is standard for A2C/PPO algorithms.
        """
        rewards = self.rewards
        values = [v.item() for v in self.values]
        dones = self.dones

        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        returns = np.zeros(num_steps)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(num_steps)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )

    def clear(self) -> None:
        """Clear all stored data."""
        self.local_obs.clear()
        self.contexts.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ── MA2C Agent ───────────────────────────────────────────────────────────────

class MA2CAgent:
    """
    Complete MA2C agent combining GAT + Actor + Critic.

    This is the main class you interact with. Each CAV in the
    simulation has one of these (or they share parameters).

    Usage:
        agent = MA2CAgent(config)
        action, log_prob, value = agent.act(local_obs, context, global_state)
        loss = agent.update(rollout)
    """

    def __init__(self, config: Optional[MA2CConfig] = None):
        self.config = config or MA2CConfig()

        # Networks
        self.gat = GATNetwork(self.config.gat_config)
        self.actor = Actor(self.config)
        self.critic = Critic(self.config)

        # Single optimiser for all parameters
        all_params = (
            list(self.gat.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.learning_rate)

        # Per-agent rollout storage
        self.rollout = RolloutStorage()

        # Training stats
        self.total_updates: int = 0
        self.training_losses: List[float] = []

    # ── Action Selection ─────────────────────────────────────────────────

    def act(
        self,
        local_obs: torch.Tensor,
        context: torch.Tensor,
        global_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action given current observations.

        During training: sample from the policy distribution.
        During execution: take the greedy action (deterministic=True).

        Args:
            local_obs:    [local_obs_dim] local observation.
            context:      [context_dim] GAT context vector.
            global_state: [global_state_dim] global info (critic only).
            deterministic: If True, take argmax action.

        Returns:
            (action_id, log_prob, value_estimate)
        """
        local_obs_batch = local_obs.unsqueeze(0)
        context_batch = context.unsqueeze(0)
        global_state_batch = global_state.unsqueeze(0)

        # Actor: get action distribution
        dist = self.actor(local_obs_batch, context_batch)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        # Critic: get value estimate
        value = self.critic(local_obs_batch, context_batch, global_state_batch)

        return action.item(), log_prob.squeeze(), value.squeeze()

    # ── Training Update ──────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Update networks using collected rollout data.

        Computes:
            - Policy gradient loss (actor)
            - Value function loss (critic)
            - Entropy bonus (for exploration)

        Returns:
            Dict with loss components.
        """
        if len(self.rollout) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        cfg = self.config

        # Compute last value for GAE
        with torch.no_grad():
            last_obs = self.rollout.local_obs[-1].unsqueeze(0)
            last_ctx = self.rollout.contexts[-1].unsqueeze(0)
            last_gs = self.rollout.global_states[-1].unsqueeze(0)
            last_value = self.critic(last_obs, last_ctx, last_gs).item()

        returns, advantages = self.rollout.compute_returns_and_advantages(
            last_value, cfg.gamma, cfg.gae_lambda
        )

        # Normalise advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack rollout data
        obs_batch = torch.stack(self.rollout.local_obs)
        ctx_batch = torch.stack(self.rollout.contexts)
        gs_batch = torch.stack(self.rollout.global_states)
        actions_batch = torch.stack(self.rollout.actions)
        old_log_probs = torch.stack(self.rollout.log_probs)

        # Forward pass
        dist = self.actor(obs_batch, ctx_batch)
        new_log_probs = dist.log_prob(actions_batch.squeeze())
        entropy = dist.entropy().mean()

        values = self.critic(obs_batch, ctx_batch, gs_batch).squeeze()

        # Policy loss (advantage-weighted log probs)
        policy_loss = -(new_log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = (
            policy_loss
            + cfg.value_loss_coeff * value_loss
            - cfg.entropy_coeff * entropy
        )

        # Optimise
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.gat.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            cfg.max_grad_norm,
        )
        self.optimizer.step()

        # Track stats
        self.total_updates += 1
        loss_val = total_loss.item()
        self.training_losses.append(loss_val)

        # Clear rollout
        self.rollout.clear()

        return {
            "policy_loss": round(policy_loss.item(), 4),
            "value_loss": round(value_loss.item(), 4),
            "entropy": round(entropy.item(), 4),
            "total_loss": round(loss_val, 4),
        }

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all network weights."""
        torch.save({
            "gat": self.gat.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str) -> None:
        """Load network weights."""
        checkpoint = torch.load(path, map_location="cpu")
        self.gat.load_state_dict(checkpoint["gat"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_updates = checkpoint.get("total_updates", 0)

    def set_eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.gat.eval()
        self.actor.eval()
        self.critic.eval()

    def set_train_mode(self) -> None:
        """Set all networks to training mode."""
        self.gat.train()
        self.actor.train()
        self.critic.train()


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MA2C Agent Test")
    print("=" * 60)

    config = MA2CConfig()
    agent = MA2CAgent(config)

    total_params = (
        sum(p.numel() for p in agent.gat.parameters())
        + sum(p.numel() for p in agent.actor.parameters())
        + sum(p.numel() for p in agent.critic.parameters())
    )
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  GAT:    {sum(p.numel() for p in agent.gat.parameters()):,}")
    print(f"  Actor:  {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic: {sum(p.numel() for p in agent.critic.parameters()):,}")

    # ── Test 1: Action selection ──
    print("\n--- Test 1: Action Selection ---")
    local_obs = torch.randn(config.local_obs_dim)
    context = torch.randn(config.context_dim)
    global_state = torch.randn(config.global_state_dim)

    action, log_prob, value = agent.act(local_obs, context, global_state)
    print(f"Action: {action} (0=follow, 1=alternative, 2=wait, 3=reroute)")
    print(f"Log prob: {log_prob.item():.4f}")
    print(f"Value estimate: {value.item():.4f}")

    # Deterministic action
    action_det, _, _ = agent.act(local_obs, context, global_state, deterministic=True)
    print(f"Deterministic action: {action_det}")

    # ── Test 2: Rollout collection ──
    print("\n--- Test 2: Rollout Collection ---")
    for step in range(config.rollout_length):
        obs = torch.randn(config.local_obs_dim)
        ctx = torch.randn(config.context_dim)
        gs = torch.randn(config.global_state_dim)

        act, lp, val = agent.act(obs, ctx, gs)

        # Simulate a reward
        reward = 1.0 if act == 0 else -0.5
        done = step == config.rollout_length - 1

        agent.rollout.add(obs, ctx, gs, torch.tensor(act), lp, reward, val, done)

    print(f"Rollout size: {len(agent.rollout)} steps")

    # ── Test 3: Training update ──
    print("\n--- Test 3: Training Update ---")
    losses = agent.update()
    print(f"Losses: {losses}")
    print(f"Total updates: {agent.total_updates}")
    print(f"Rollout cleared: {len(agent.rollout) == 0}")

    # ── Test 4: Multiple updates ──
    print("\n--- Test 4: Multiple Training Updates ---")
    for epoch in range(5):
        # Collect rollout
        for step in range(config.rollout_length):
            obs = torch.randn(config.local_obs_dim)
            ctx = torch.randn(config.context_dim)
            gs = torch.randn(config.global_state_dim)
            act, lp, val = agent.act(obs, ctx, gs)
            reward = float(np.random.randn())
            agent.rollout.add(obs, ctx, gs, torch.tensor(act), lp, reward, val, False)

        losses = agent.update()
        print(f"  Epoch {epoch+1}: total_loss={losses['total_loss']:.4f} "
              f"policy={losses['policy_loss']:.4f} "
              f"value={losses['value_loss']:.4f} "
              f"entropy={losses['entropy']:.4f}")

    # ── Test 5: Save/Load ──
    print("\n--- Test 5: Save/Load ---")
    agent.save("/tmp/ma2c_test.pt")
    agent2 = MA2CAgent(config)
    agent2.load("/tmp/ma2c_test.pt")
    print(f"Loaded. Updates: {agent2.total_updates}")

    # Verify same output
    with torch.no_grad():
        act1, _, val1 = agent.act(local_obs, context, global_state, deterministic=True)
        act2, _, val2 = agent2.act(local_obs, context, global_state, deterministic=True)
    print(f"Original action={act1}, value={val1.item():.4f}")
    print(f"Loaded action={act2}, value={val2.item():.4f}")
    print(f"Match: {act1 == act2}")

    # ── Test 6: Global state builder ──
    print("\n--- Test 6: Global State Builder ---")
    gs = GlobalStateBuilder.build(
        total_vehicles=100, avg_link_density=0.4, max_link_density=0.9,
        num_obstacles=2, avg_system_speed=0.7, cav_fraction=0.6,
        num_clusters=5, avg_cluster_size=12.0, total_trips_completed=500,
        avg_mstt=8.5, num_stalled=3, total_recalculations=50,
    )
    print(f"Global state (16 features): {[round(x, 3) for x in gs]}")

    print("\nAll tests passed.")