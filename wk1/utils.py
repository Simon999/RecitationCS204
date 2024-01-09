import networkx as nx
import random

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