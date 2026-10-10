"""Bus-eligible road graph representation and routing utilities."""

import math

import networkx as nx
import pandas as pd

from jro.config import VehicleProfile
from jro.logging_setup import logger


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two WGS-84 coordinates in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


ROAD_SPEED_LIMITS_KMH: dict[str, float] = {
    "trunk": 50.0,
    "primary": 40.0,
    "secondary": 30.0,
    "tertiary": 25.0,
    "unclassified": 20.0,
    "residential": 15.0,
}


class RoadNetwork:
    """Directed graph representing navigable road network for bus operations."""

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.graph = graph or nx.DiGraph()

    @classmethod
    def build_from_nodes_and_edges(
        cls,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        profile: VehicleProfile,
    ) -> "RoadNetwork":
        """Builds directed graph filtering edges by vehicle profile allowed highway classes.

        Args:
            nodes_df: DataFrame with ['node_id', 'latitude', 'longitude'].
            edges_df: DataFrame with ['u', 'v', 'highway', 'length_km', 'oneway'].
            profile: Target VehicleProfile.

        Returns:
            Configured RoadNetwork instance.
        """
        allowed = set(profile.allowed_highway)
        g = nx.DiGraph()

        # Add nodes
        for _, row in nodes_df.iterrows():
            nid = str(row["node_id"])
            g.add_node(nid, lat=float(row["latitude"]), lon=float(row["longitude"]))

        # Add edges
        for _, row in edges_df.iterrows():
            hw = str(row["highway"])
            if hw not in allowed:
                continue

            u, v = str(row["u"]), str(row["v"])
            if u not in g or v not in g:
                continue

            length_km = float(row["length_km"])
            speed_kmh = ROAD_SPEED_LIMITS_KMH.get(hw, 25.0)
            ff_time_min = (length_km / speed_kmh) * 60.0

            g.add_edge(
                u,
                v,
                highway=hw,
                length_km=length_km,
                speed_kmh=speed_kmh,
                free_flow_min=ff_time_min,
                weight=ff_time_min,
            )

            # Two-way if not oneway
            is_oneway = bool(row.get("oneway", False))
            if not is_oneway and not g.has_edge(v, u):
                g.add_edge(
                    v,
                    u,
                    highway=hw,
                    length_km=length_km,
                    speed_kmh=speed_kmh,
                    free_flow_min=ff_time_min,
                    weight=ff_time_min,
                )

        logger.info(
            "Built RoadNetwork for %s: %d nodes, %d edges",
            profile.name,
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return cls(graph=g)

    def find_nearest_node(self, lat: float, lon: float) -> str | None:
        """Finds closest graph node to a given coordinate."""
        if self.graph.number_of_nodes() == 0:
            return None

        best_node = None
        min_dist = float("inf")
        for node, data in self.graph.nodes(data=True):
            nlat, nlon = data["lat"], data["lon"]
            d = (lat - nlat) ** 2 + (lon - nlon) ** 2
            if d < min_dist:
                min_dist = d
                best_node = node
        return best_node

    def shortest_path(self, source: str, target: str, weight: str = "weight") -> list[str] | None:
        """Computes shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def path_length_km(self, path: list[str]) -> float:
        """Calculates total route length in kilometers along path nodes."""
        total_km = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                total_km += self.graph[u][v].get("length_km", 0.0)
            else:
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                total_km += haversine_distance_km(
                    u_data["lat"], u_data["lon"], v_data["lat"], v_data["lon"]
                )
        return total_km
