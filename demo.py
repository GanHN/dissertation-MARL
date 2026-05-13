"""
demo_video.py
Short controlled demonstration 
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.environment.grid_network import GridNetwork, NetworkConfig
from src.environment.vehicle import CAV, VehicleState
from src.communication.comm_manager import CommunicationManager, CommConfig
from src.routing.dec_ctdsp import dec_ctdsp_route
from src.marl.gat_network import ObservationBuilder, build_cluster_graph
from src.marl.ma2c import MA2CAgent, MA2CConfig

ACTION_NAMES = {
    0: "follow current Dec-CTDSP route",
    1: "use fallback alternative route",
    2: "wait one timestep",
    3: "reroute with Dec-CTDSP",
}


def route_text(route):
    return " -> ".join(str(n) for n in route) if route else "<no route>"


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def build_demo_cavs(network: GridNetwork):
    """Create a tiny controlled CAV cluster."""
    cav0 = CAV(vehicle_id=0, origin=(2, 0), destination=(2, 5), blacklist_ttl=40)
    cav1 = CAV(vehicle_id=1, origin=(1, 0), destination=(1, 5), blacklist_ttl=40)
    cav2 = CAV(vehicle_id=2, origin=(3, 0), destination=(3, 5), blacklist_ttl=40)

    cav0.current_node = (2, 1)
    cav1.current_node = (1, 1)
    cav2.current_node = (3, 1)

    for cav in (cav0, cav1, cav2):
        cav.state = VehicleState.EN_ROUTE
        cav.speed = 1.0
        cav.set_routing_function(dec_ctdsp_route)
        cav.compute_route(network, timestep=0, cluster_vehicles=[])

    return [cav0, cav1, cav2]


def demo_omm_and_rerouting() -> tuple[GridNetwork, list[CAV], CommunicationManager]:
    print_header("Controlled demo: OMM obstacle memory + Dec-CTDSP rerouting")

    network = GridNetwork(NetworkConfig(rows=6, cols=6, link_capacity=5))
    vehicles = build_demo_cavs(network)
    cav0, cav1, cav2 = vehicles

    comm = CommunicationManager(CommConfig(communication_radius=1.5, enable_multi_hop=True))
    comm.update_clusters(vehicles, network)

    print("Initial CAV positions:")
    for cav in vehicles:
        print(f"  CAV_{cav.vehicle_id}: node={cav.current_node}, destination={cav.destination}")

    print("\nCommunication clusters with CR=1.5:")
    print(f"  {comm.clusters}")

    print("\nInitial route for CAV_0 before any obstacle:")
    print(f"  {route_text([cav0.current_node] + cav0.planned_route)}")

    # Controlled obstacle directly on CAV_0's current route.
    blocked_node = (2, 2)
    network.block_node(blocked_node)
    t_detect = 5

    print(f"\nTimestep {t_detect}: controlled obstacle becomes active at node {blocked_node}")
    print(f"CAV_0 next planned node is {cav0.get_next_node()}")

    if cav0.get_next_node() == blocked_node:
        print(f"CAV_0 detects that its next planned node {blocked_node} is blocked.")
    else:
        print("CAV_0 route does not point directly at the obstacle, but we broadcast it for the demo.")

    receivers = comm.broadcast_obstacle(cav0, blocked_node, vehicles, timestep=t_detect)
    print(f"\nOMM broadcast:")
    print(f"  Sender: CAV_0")
    print(f"  Blocked node: {blocked_node}")
    print(f"  Receivers in same cluster: {receivers}")

    print("\nBlacklists immediately after broadcast:")
    for cav in vehicles:
        print(f"  CAV_{cav.vehicle_id}: {sorted(cav.get_blacklisted_nodes())}")

    cluster_for_cav0 = comm.get_cluster_members(cav0.vehicle_id, vehicles, exclude_self=True)
    old_route = list(cav0.planned_route)
    cav0.compute_route(network, timestep=t_detect, cluster_vehicles=cluster_for_cav0)
    new_route = list(cav0.planned_route)

    print("\nDec-CTDSP reroute using the updated OMM blacklist:")
    print(f"  Old route: {route_text([cav0.current_node] + old_route)}")
    print(f"  New route: {route_text([cav0.current_node] + new_route)}")
    print(f"  Does the new route avoid {blocked_node}? {blocked_node not in new_route}")

    # Confirmation example: neighbouring CAVs can refresh the TTL while the node remains blocked.
    t_confirm = 15
    confirmations = comm.propagate_confirmations(vehicles, network, timestep=t_confirm)
    print(f"\nTimestep {t_confirm}: confirmation-based persistence")
    print(f"  Confirmations propagated: {confirmations}")
    if blocked_node in cav0.blacklist:
        entry = cav0.blacklist[blocked_node]
        print(f"  CAV_0 last_confirmed for {blocked_node}: {entry.last_confirmed}")

    # obstacle clears, and without future confirmations the blacklist entry expires.
    t_clear = 45
    network.unblock_node(blocked_node)
    print(f"\nTimestep {t_clear}: obstacle clears in the environment")
    print("  OMM does not instantly delete memory; it waits for TTL expiry.")

    t_expire = 56  # 56 - last_confirmed(15) = 41 > TTL(40)
    expired = comm.decay_all_blacklists(vehicles, timestep=t_expire)
    print(f"\nTimestep {t_expire}: TTL decay check")
    print(f"  Expired blacklist entries: {expired}")
    print("  Blacklists after TTL decay:")
    for cav in vehicles:
        print(f"    CAV_{cav.vehicle_id}: {sorted(cav.get_blacklisted_nodes())}")

    return network, vehicles, comm


def demo_ma2c_action(network: GridNetwork, vehicles: list[CAV], comm: CommunicationManager) -> None:
    print_header("demo: trained MA2C/GAT action probabilities")

    checkpoint = ROOT / "results" / "marl" / "final_model.pt"
    if not checkpoint.exists():
        print("No trained checkpoint found at results/marl/final_model.pt")
        print("Skipping the MA2C part. The OMM + Dec-CTDSP demo above is still enough for the video.")
        return

    agent = MA2CAgent(MA2CConfig())
    gat_loaded = True
    try:
        agent.load(str(checkpoint))
    except RuntimeError as exc:
        print("Checkpoint found, but full GAT weights could not be loaded on this machine.")
        print("This usually means PyTorch Geometric is missing or different from training.")
        print("Loading the trained actor only and using zero GAT context for the probability demo.")
        checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
        agent.actor.load_state_dict(checkpoint_data["actor"])
        gat_loaded = False
    agent.set_eval_mode()

    # Build one GAT input tensor from the three CAVs.
    node_features = []
    for cav in vehicles:
        remaining = len(cav.get_remaining_route())
        obs = ObservationBuilder.build_local_obs(
            current_node=cav.current_node,
            destination=cav.destination,
            speed=cav.speed,
            dist_remaining=float(remaining),
            time_elapsed=5.0,
            num_blacklisted=len(cav.get_blacklisted_nodes()),
            grid_rows=6,
            grid_cols=6,
        )
        node_features.append(obs)

    node_features_t = torch.tensor(node_features, dtype=torch.float32)
    edge_index = build_cluster_graph(list(range(len(vehicles))))

    with torch.no_grad():
        if gat_loaded:
            contexts = agent.gat(node_features_t, edge_index)
            cav0_ctx = contexts[0].unsqueeze(0)
        else:
            cav0_ctx = torch.zeros((1, agent.config.context_dim), dtype=torch.float32)
        cav0_obs = node_features_t[0].unsqueeze(0)
        dist = agent.actor(cav0_obs, cav0_ctx)
        probs = dist.probs.squeeze(0)
        selected = int(torch.argmax(probs).item())

    print("Using saved checkpoint: results/marl/final_model.pt")
    if gat_loaded:
        print("CAV_0 local observation + GAT context were passed into the actor.")
    else:
        print("CAV_0 local observation + zero context were passed into the trained actor.")
        print("On your own machine with PyG installed, this should load the full GAT + actor checkpoint.")
    print("\nAction probabilities:")
    for action_id, p in enumerate(probs.tolist()):
        print(f"  action {action_id} ({ACTION_NAMES[action_id]}): {p:.3f}")
    print(f"\nSelected deterministic action: {selected} -> {ACTION_NAMES[selected]}")


def main() -> None:
    print("Video demonstration script for Dec-CTDSP + OMM + MA2C")
    print("This is a short controlled demo, not a full training or benchmark run.")

    network, vehicles, comm = demo_omm_and_rerouting()
    demo_ma2c_action(network, vehicles, comm)

    print_header("What this demonstrates")
    print("1. CAVs form a communication cluster.")
    print("2. A blocked node is detected and broadcast through OMM.")
    print("3. Nearby CAVs update their own blacklists.")
    print("4. Dec-CTDSP recomputes a route that avoids the blacklisted node.")
    print("5. TTL decay removes stale obstacle memory after the obstacle is no longer confirmed.")
    print("6. If the trained checkpoint is present, MA2C/GAT action probabilities are shown.")


if __name__ == "__main__":
    main()
