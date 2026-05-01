"""
vehicle.py - Vehicle Agents for CAV Simulation
Defines the vehicle types that operate on the grid network.
    CAV             - communication, OMM blacklist, Dec-CTDSP routing
    HDV             - predefined naive routing (no communication)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from src.environment.grid_network import GridNetwork



class VehicleType(Enum):
    CAV = auto()
    HDV = auto()


class VehicleState(Enum):
    WAITING = auto()        # At origin, not yet departed or waiting to re-depart
    EN_ROUTE = auto()       # Travelling on the network
    AT_INTERSECTION = auto()  # Arrived at an intersection, deciding next move
    ARRIVED = auto()        # Reached destination




@dataclass
class MobilityMessage:
    """
    Lightweight message broadcasted by a CAV to its communication cluster.
    This is ~12-16 bytes: vehicle_id + node + speed + route hash.
    """
    vehicle_id: int
    current_node: Tuple[int, int]
    speed: float
    planned_route: List[Tuple[int, int]]
    timestamp: int  # Simulation timestep when the message was created



@dataclass
class BlacklistEntry:
    """
    A single entry in the OMM blacklist.

    Uses confirmation-based persistence: the entry has a TTL (time-to-live)
    that gets refreshed whenever another CAV reconfirms the blockage.
    If no confirmation is received within the TTL, the entry expires
    and the node is removed from the blacklist.
    """
    node: Tuple[int, int]
    added_at: int           # Timestep when first added
    last_confirmed: int     # Timestep of most recent confirmation
    ttl: int                # Time-to-live in timesteps



class Vehicle:
    """
    Base class for all vehicles in the simulation.

    Tracks position, destination, planned route, speed, and trip statistics.
    Subclasses (CAV, HDV) implement different routing strategies.
    """

    def __init__(
        self,
        vehicle_id: int,
        origin: Tuple[int, int],
        destination: Tuple[int, int],
        vehicle_type: VehicleType,
    ):
        # Identity
        self.vehicle_id: int = vehicle_id
        self.vehicle_type: VehicleType = vehicle_type

        # Trip endpoints
        self.origin: Tuple[int, int] = origin
        self.destination: Tuple[int, int] = destination

        # Current state
        self.current_node: Tuple[int, int] = origin
        self.state: VehicleState = VehicleState.WAITING
        self.speed: float = 0.0
        # Continuous along-link state:
        # None => at an intersection (current_node).
        # Otherwise moving from link_start_node to link_end_node with progress in [0, 1).
        self.link_start_node: Optional[Tuple[int, int]] = None
        self.link_end_node: Optional[Tuple[int, int]] = None
        self.link_progress: float = 0.0

        # Route: ordered list of nodes from current position to destination
        self.planned_route: List[Tuple[int, int]] = []
        self.route_index: int = 0  # Index into planned_route for next node

        # Travel statistics
        self.departure_time: Optional[int] = None
        self.arrival_time: Optional[int] = None
        self.total_travel_time: float = 0.0
        self.total_wait_time: float = 0.0
        self.num_route_recalculations: int = 0
        self.trips_completed: int = 0

        # History of travel times (for averaging across trips)
        self.trip_travel_times: List[float] = []


    def set_route(self, route: List[Tuple[int, int]]) -> None:
        """
        Set a new planned route for this vehicle.

        Args:
            route: Ordered list of nodes from current position to destination.
                   Should NOT include the current node as the first element.
        """
        self.planned_route = route
        self.route_index = 0

    def get_next_node(self) -> Optional[Tuple[int, int]]:
        """Return the next node on the planned route, or None if at end."""
        if self.route_index < len(self.planned_route):
            return self.planned_route[self.route_index]
        return None

    def advance_route(self) -> None:
        """Move the route pointer forward after reaching the next node."""
        if self.route_index < len(self.planned_route):
            self.current_node = self.planned_route[self.route_index]
            self.route_index += 1

    def has_reached_destination(self) -> bool:
        """Check if the vehicle is at its destination."""
        return self.current_node == self.destination

    def get_remaining_route(self) -> List[Tuple[int, int]]:
        """Return the portion of the route not yet traversed."""
        return self.planned_route[self.route_index:]

    def is_at_intersection(self) -> bool:
        """True when the vehicle is exactly at an intersection."""
        return self.link_end_node is None

    def start_link_traversal(self, next_node: Tuple[int, int]) -> None:
        """Start traversing from current_node toward next_node."""
        self.link_start_node = self.current_node
        self.link_end_node = next_node
        self.link_progress = 0.0

    def clear_link_traversal(self) -> None:
        """Clear along-link state (vehicle is at an intersection)."""
        self.link_start_node = None
        self.link_end_node = None
        self.link_progress = 0.0

    def move_along_current_link(self, distance_blocks: float, block_length: float = 1.0) -> bool:
        """
        Advance along the current link by a continuous distance.

        Returns:
            True if the vehicle reached the next intersection this call.
        """
        if self.link_end_node is None:
            return False
        if block_length <= 1e-8:
            block_length = 1.0

        delta = max(0.0, distance_blocks) / block_length
        self.link_progress += delta

        if self.link_progress >= 1.0:
            self.current_node = self.link_end_node
            self.route_index += 1
            self.clear_link_traversal()
            return True
        return False

    def get_continuous_position(self, network: GridNetwork) -> Tuple[float, float]:
        """
        Get continuous (x, y) position for communication/range checks.
        """
        if self.link_end_node is None or self.link_start_node is None:
            return network.get_node_position(self.current_node)

        x0, y0 = network.get_node_position(self.link_start_node)
        x1, y1 = network.get_node_position(self.link_end_node)
        p = min(max(self.link_progress, 0.0), 1.0)
        return (x0 + (x1 - x0) * p, y0 + (y1 - y0) * p)


    def depart(self, timestep: int) -> None:
        """Start (or restart) a trip from the origin."""
        self.state = VehicleState.EN_ROUTE
        self.departure_time = timestep
        self.arrival_time = None
        self.total_travel_time = 0.0
        self.total_wait_time = 0.0
        self.clear_link_traversal()

    def arrive(self, timestep: int) -> None:
        """Record arrival at the destination."""
        self.state = VehicleState.ARRIVED
        self.arrival_time = timestep
        if self.departure_time is not None:
            trip_time = timestep - self.departure_time
            self.trip_travel_times.append(trip_time)
        self.trips_completed += 1

    def reset_for_new_trip(self, new_destination: Tuple[int, int]) -> None:
        """
        Reset the vehicle for a new trip (continuous simulation).
        """
        self.current_node = self.origin
        self.destination = new_destination
        self.state = VehicleState.WAITING
        self.speed = 0.0
        self.planned_route = []
        self.route_index = 0
        self.departure_time = None
        self.arrival_time = None
        self.num_route_recalculations = 0
        self.clear_link_traversal()


    def compute_route(self, network: GridNetwork, timestep: int) -> None:
        """
        Compute a route from current position to destination.

        This is the method where CAV and HDV behaviour diverges.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute_route()")


    def get_average_travel_time(self) -> float:
        """Return average trip time across all completed trips."""
        if not self.trip_travel_times:
            return 0.0
        return sum(self.trip_travel_times) / len(self.trip_travel_times)

    def __repr__(self) -> str:
        return (
            f"{self.vehicle_type.name}(id={self.vehicle_id}, "
            f"pos={self.current_node}, dest={self.destination}, "
            f"state={self.state.name})"
        )



class HDV(Vehicle):
    """
    Human-Driven Vehicle with predefined naive routing.
    HDVs do NOT communicate and do NOT have an OMM blacklist.
    They are completely unaware of obstacles until they physically
    encounter one (at which point they simply wait).
    """

    def __init__(
        self,
        vehicle_id: int,
        origin: Tuple[int, int],
        destination: Tuple[int, int],
    ):
        super().__init__(vehicle_id, origin, destination, VehicleType.HDV)

    def compute_route(self, network: GridNetwork, timestep: int) -> None:
        """
        Compute the predefined naive route (Equation 1 from paper).

        The route is: go east some steps, then go north/south to align
        with the destination row, then go east the remaining steps.

        On this grid:
            - East  = (row, col) -> (row, col+1)
            - North = (row, col) -> (row-1, col)  (row decreases = up)
            - South = (row, col) -> (row+1, col)  (row increases = down)
        """
        route = []
        r, c = self.current_node
        dest_r, dest_c = self.destination

        dy = dest_r - r  # Positive = need to go south, negative = north
        total_east_steps = dest_c - c

        if total_east_steps < 0:
            # Edge case: can't go west on one-way eastbound streets
            # This shouldn't happen in normal simulation, but handle gracefully
            self.planned_route = []
            self.route_index = 0
            return

        # Decide how many east steps before vertical adjustment
        # Paper formula: go east for (total_east_steps - |dy|) steps first,
        # then vertical, then remaining east steps.
        # But we need to ensure the split makes sense.
        east_before_vertical = max(0, total_east_steps - abs(dy))

        # Phase 1: Go east
        current_r, current_c = r, c
        for _ in range(east_before_vertical):
            current_c += 1
            route.append((current_r, current_c))

        # Phase 2: Go north or south
        if dy > 0:
            # Need to go south (increase row)
            for _ in range(abs(dy)):
                current_r += 1
                route.append((current_r, current_c))
        elif dy < 0:
            # Need to go north (decrease row)
            for _ in range(abs(dy)):
                current_r -= 1
                route.append((current_r, current_c))

        # Phase 3: Go east for remaining steps
        remaining_east = dest_c - current_c
        for _ in range(remaining_east):
            current_c += 1
            route.append((current_r, current_c))

        self.set_route(route)



class CAV(Vehicle):
    """
    Connected Autonomous Vehicle with communication and OMM.

    Key capabilities:
        - Broadcasts MobilityMessages to neighbours within comm radius
        - Maintains an OMM blacklist with confirmation-based TTL
        - Routes via Dec-CTDSP (to be wired in from src/routing/)
        - Falls back to random routing if no cluster neighbours exist

    The actual Dec-CTDSP routing is delegated to an external function
    that will be set via set_routing_function(). This keeps vehicle.py
    decoupled from the routing module.
    """

    DEFAULT_BLACKLIST_TTL: int = 50  # Timesteps before unconfirmed entry expires

    def __init__(
        self,
        vehicle_id: int,
        origin: Tuple[int, int],
        destination: Tuple[int, int],
        blacklist_ttl: int = DEFAULT_BLACKLIST_TTL,
    ):
        super().__init__(vehicle_id, origin, destination, VehicleType.CAV)

        # OMM Blacklist: node -> BlacklistEntry
        self.blacklist: Dict[Tuple[int, int], BlacklistEntry] = {}
        self.blacklist_ttl: int = blacklist_ttl

        # External routing function (injected by the simulator)
        # Signature: (network, current_node, destination, blacklist_nodes) -> route
        self._routing_function = None


    def set_routing_function(self, func) -> None:
        """
        Set the external routing function (e.g., Dec-CTDSP).

        Args:
            func: Callable with signature:
                  (network: GridNetwork,
                   source: Tuple[int, int],
                   target: Tuple[int, int],
                   blacklist: Set[Tuple[int, int]],
                   cluster_vehicles: List[Vehicle],
                   timestep: int) -> List[Tuple[int, int]]
        """
        self._routing_function = func


    def add_to_blacklist(self, node: Tuple[int, int], timestep: int) -> None:
        """
        Add a blocked node to the OMM blacklist or refresh its TTL.
        If the node is already blacklisted, update last_confirmed.
        """
        if node in self.blacklist:
            # Refresh: another vehicle confirmed the blockage still exists
            self.blacklist[node].last_confirmed = timestep
        else:
            self.blacklist[node] = BlacklistEntry(
                node=node,
                added_at=timestep,
                last_confirmed=timestep,
                ttl=self.blacklist_ttl,
            )

    def decay_blacklist(self, timestep: int) -> List[Tuple[int, int]]:
        """
        Remove expired entries from the blacklist.

        An entry expires if (timestep - last_confirmed) > ttl.
        This implements your confirmation-based persistence design.

        Returns:
            List of nodes that were removed (expired).
        """
        expired = []
        for node, entry in list(self.blacklist.items()):
            if (timestep - entry.last_confirmed) > entry.ttl:
                expired.append(node)
                del self.blacklist[node]
        return expired

    def get_blacklisted_nodes(self) -> Set[Tuple[int, int]]:
        """Return the set of currently blacklisted node IDs."""
        return set(self.blacklist.keys())

    def receive_obstacle_broadcast(
        self,
        blocked_node: Tuple[int, int],
        timestep: int
    ) -> None:
        """
        Handle an incoming obstacle broadcast from another CAV.
        """
        self.add_to_blacklist(blocked_node, timestep)


    def create_mobility_message(self, timestep: int) -> MobilityMessage:
        """
        Create a mobility message to broadcast to the communication cluster.

        Contains: vehicle ID, current location, speed, and planned route.
        """
        return MobilityMessage(
            vehicle_id=self.vehicle_id,
            current_node=self.current_node,
            speed=self.speed,
            planned_route=list(self.planned_route),
            timestamp=timestep,
        )


    def compute_route(
        self,
        network: GridNetwork,
        timestep: int,
        cluster_vehicles: Optional[List[Vehicle]] = None,
    ) -> None:
        """
        Compute route using Dec-CTDSP or fall back to random valid path.
        Args:
            network:           The grid network.
            timestep:          Current simulation timestep.
            cluster_vehicles:  Other CAVs in the communication cluster.
                               If None or empty, falls back to random routing.
        """
        # First, decay expired blacklist entries
        self.decay_blacklist(timestep)
        blacklisted = self.get_blacklisted_nodes()

        if self._routing_function is not None:
            # Use Dec-CTDSP (or whatever routing was injected)
            route = self._routing_function(
                network,
                self.current_node,
                self.destination,
                blacklisted,
                cluster_vehicles or [],
                timestep,
            )
            if route:
                self.set_route(route)
                self.num_route_recalculations += 1
                return

        # Fallback: simple Dijkstra on static weights, excluding blacklisted nodes
        route = self._fallback_dijkstra(network, blacklisted)
        if route:
            self.set_route(route)
            self.num_route_recalculations += 1

    def _fallback_dijkstra(
        self,
        network: GridNetwork,
        blacklisted: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Simple static Dijkstra as fallback when no cluster exists.

        Excludes blacklisted nodes from the search. Uses free-flow
        travel times as edge weights (no time-dependency).
        """
        import heapq

        source = self.current_node
        target = self.destination

        # Priority queue: (cumulative_cost, node)
        pq = [(0.0, source)]
        dist = {source: 0.0}
        prev = {source: None}

        while pq:
            cost, u = heapq.heappop(pq)

            if u == target:
                # Reconstruct path (excluding the source node)
                path = []
                node = target
                while node is not None and node != source:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                return path

            if cost > dist.get(u, float("inf")):
                continue

            for v in network.get_neighbors(u):
                # Skip blacklisted nodes (but always allow the destination)
                if v in blacklisted and v != target:
                    continue

                edge_tt = network.get_free_flow_travel_time(u, v)
                new_cost = cost + edge_tt

                if new_cost < dist.get(v, float("inf")):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(pq, (new_cost, v))

        # No path found
        return []


    def reset_for_new_trip(self, new_destination: Tuple[int, int]) -> None:
        """Reset for a new trip. Blacklist persists across trips."""
        super().reset_for_new_trip(new_destination)
        # Note: blacklist is NOT cleared — it carries over between trips.
        # Entries still expire naturally via TTL decay.



class VehicleFactory:
    """
    Creates a mixed fleet of CAVs and HDVs based on Market Penetration.
    """

    @staticmethod
    def create_fleet(
        num_vehicles: int,
        market_penetration: float,
        origins: List[Tuple[int, int]],
        destinations: List[Tuple[int, int]],
        blacklist_ttl: int = CAV.DEFAULT_BLACKLIST_TTL,
        seed: Optional[int] = None,
    ) -> List[Vehicle]:
        """
        Create a mixed fleet of vehicles.
        Args:
            num_vehicles:       Total number of vehicles.
            market_penetration: Fraction of vehicles that are CAVs (0.0 to 1.0).
            origins:            List of origin nodes to distribute vehicles across.
            destinations:       List of destination nodes to assign randomly.
            blacklist_ttl:      TTL for CAV blacklist entries.
            seed:               Random seed for reproducibility.

        Returns:
            List of Vehicle objects (mix of CAV and HDV).
        """
        if seed is not None:
            random.seed(seed)

        num_cavs = int(num_vehicles * market_penetration)
        num_hdvs = num_vehicles - num_cavs

        vehicles: List[Vehicle] = []
        vehicle_id = 0

        # Create CAVs
        for _ in range(num_cavs):
            origin = origins[vehicle_id % len(origins)]
            dest = random.choice(destinations)
            cav = CAV(
                vehicle_id=vehicle_id,
                origin=origin,
                destination=dest,
                blacklist_ttl=blacklist_ttl,
            )
            vehicles.append(cav)
            vehicle_id += 1

        # Create HDVs
        for _ in range(num_hdvs):
            origin = origins[vehicle_id % len(origins)]
            dest = random.choice(destinations)
            hdv = HDV(
                vehicle_id=vehicle_id,
                origin=origin,
                destination=dest,
            )
            vehicles.append(hdv)
            vehicle_id += 1

        # Shuffle so CAVs and HDVs are interleaved (not grouped)
        random.shuffle(vehicles)
        return vehicles

    @staticmethod
    def reassign_destination(
        vehicle: Vehicle,
        destinations: List[Tuple[int, int]],
    ) -> None:
        """Assign a new random destination for a new trip."""
        new_dest = random.choice(destinations)
        vehicle.reset_for_new_trip(new_dest)



if __name__ == "__main__":
    # We need to add parent to path for the import to work standalone
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.environment.grid_network import GridNetwork, NetworkConfig

    # Build network
    config = NetworkConfig()
    network = GridNetwork(config)

    print("=" * 60)
    print("Vehicle Module Test")
    print("=" * 60)

    # ── Test HDV routing ──
    print("\n--- HDV Routing Test (Equation 1) ---")
    hdv = HDV(vehicle_id=0, origin=(0, 0), destination=(3, 5))
    hdv.compute_route(network, timestep=0)
    print(f"HDV: {hdv.origin} -> {hdv.destination}")
    print(f"Route: {hdv.planned_route}")
    print(f"Route length: {len(hdv.planned_route)} steps")

    hdv2 = HDV(vehicle_id=1, origin=(4, 0), destination=(1, 5))
    hdv2.compute_route(network, timestep=0)
    print(f"\nHDV: {hdv2.origin} -> {hdv2.destination}")
    print(f"Route: {hdv2.planned_route}")
    print(f"Route length: {len(hdv2.planned_route)} steps")

    # Same row (no vertical movement)
    hdv3 = HDV(vehicle_id=2, origin=(2, 0), destination=(2, 5))
    hdv3.compute_route(network, timestep=0)
    print(f"\nHDV: {hdv3.origin} -> {hdv3.destination}")
    print(f"Route: {hdv3.planned_route}")
    print(f"Route length: {len(hdv3.planned_route)} steps")

    # ── Test CAV with blacklist ──
    print("\n--- CAV Blacklist Test ---")
    cav = CAV(vehicle_id=10, origin=(0, 0), destination=(2, 5), blacklist_ttl=10)

    # Add blocked node
    cav.add_to_blacklist((1, 2), timestep=0)
    cav.add_to_blacklist((2, 3), timestep=0)
    print(f"Blacklisted nodes: {cav.get_blacklisted_nodes()}")

    # Compute route (uses fallback Dijkstra since no routing function set)
    cav.compute_route(network, timestep=1)
    print(f"CAV route avoiding blacklisted: {cav.planned_route}")
    print(f"Route avoids (1,2)? {(1, 2) not in cav.planned_route}")
    print(f"Route avoids (2,3)? {(2, 3) not in cav.planned_route}")

    # Test TTL decay
    expired = cav.decay_blacklist(timestep=15)  # TTL=10, added at t=0
    print(f"\nAfter decay at t=15: expired={expired}")
    print(f"Remaining blacklist: {cav.get_blacklisted_nodes()}")

    # Test confirmation refresh
    cav2 = CAV(vehicle_id=11, origin=(1, 0), destination=(3, 5), blacklist_ttl=10)
    cav2.add_to_blacklist((2, 2), timestep=0)
    cav2.add_to_blacklist((2, 2), timestep=8)  # Refresh at t=8
    expired2 = cav2.decay_blacklist(timestep=15)  # 15 - 8 = 7 < TTL of 10
    print(f"\nConfirmation test: refreshed at t=8, decay at t=15")
    print(f"Expired: {expired2}")
    print(f"Still blacklisted: {cav2.get_blacklisted_nodes()}")

    # ── Test MobilityMessage ──
    print("\n--- Mobility Message Test ---")
    cav.speed = 0.8
    msg = cav.create_mobility_message(timestep=5)
    print(f"Message: vehicle={msg.vehicle_id}, node={msg.current_node}, "
          f"speed={msg.speed}, route_len={len(msg.planned_route)}")

    # ── Test Fleet Creation ──
    print("\n--- Fleet Creation Test ---")
    fleet = VehicleFactory.create_fleet(
        num_vehicles=20,
        market_penetration=0.6,
        origins=network.origins,
        destinations=network.destinations,
        seed=42,
    )
    num_cavs = sum(1 for v in fleet if v.vehicle_type == VehicleType.CAV)
    num_hdvs = sum(1 for v in fleet if v.vehicle_type == VehicleType.HDV)
    print(f"Fleet: {len(fleet)} vehicles ({num_cavs} CAVs, {num_hdvs} HDVs)")
    print(f"MP = {num_cavs / len(fleet):.0%}")

    # Compute routes for all HDVs and check they're valid
    for v in fleet:
        if isinstance(v, HDV):
            v.compute_route(network, timestep=0)
    hdv_with_routes = [v for v in fleet if isinstance(v, HDV) and v.planned_route]
    print(f"HDVs with valid routes: {len(hdv_with_routes)}/{num_hdvs}")

    # Compute routes for all CAVs (fallback Dijkstra)
    for v in fleet:
        if isinstance(v, CAV):
            v.compute_route(network, timestep=0)
    cav_with_routes = [v for v in fleet if isinstance(v, CAV) and v.planned_route]
    print(f"CAVs with valid routes: {len(cav_with_routes)}/{num_cavs}")

    # ── Test trip lifecycle ──
    print("\n--- Trip Lifecycle Test ---")
    test_v = fleet[0]
    test_v.compute_route(network, timestep=0) if not test_v.planned_route else None
    test_v.depart(timestep=0)
    print(f"Departed: {test_v}")

    # Simulate moving through route
    while test_v.get_next_node() is not None:
        test_v.advance_route()
    test_v.arrive(timestep=5)
    print(f"Arrived: {test_v}")
    print(f"Trip travel time: {test_v.trip_travel_times}")

    # Reset for new trip
    VehicleFactory.reassign_destination(test_v, network.destinations)
    print(f"Reset for new trip: {test_v}")
    print(f"Trips completed: {test_v.trips_completed}")
