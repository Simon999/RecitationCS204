import networkx as nx
from utils import *
from search import *


# generate graph
G = get_rand_connected_graph(8,0.3)
pos = nx.spring_layout(G)


# test Uninformed Search
start = get_center_vertex(G)
dfs(G,start, pos)
bfs(G,start, pos)
ucs(G,start,pos)


# test Informed Search
start, goal = get_farthest_vertices(G)
greedy_search(G, start, goal, pos, euclidean_distance)
a_star_search(G, start, goal, pos, euclidean_distance)