import networkx as nx
import random
from typing import Tuple

def get_rand_connected_graph(n: int, p: float) -> nx.Graph:
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            for (u, v) in G.edges():
                G[u][v]['weight'] = random.randint(1, 10)
            return G

def get_center_vertex(G: nx.Graph) -> int:
    centrality = nx.closeness_centrality(G)
    most_centered = max(centrality, key=centrality.get)
    return most_centered

def get_farthest_vertices(G: nx.Graph) -> Tuple[int, int]:
    max_distance = 0
    farthest_vertices = (None, None)

    for node1 in G.nodes:
        distances = nx.single_source_shortest_path_length(G, node1)
        for node2, distance in distances.items():
            if distance > max_distance:
                max_distance = distance
                farthest_vertices = (node1, node2)

    return farthest_vertices

def euclidean_distance(pos, u: int, v: int):
    return ((pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2) ** 0.5