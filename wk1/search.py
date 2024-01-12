import networkx as nx
import matplotlib.pyplot as plt
import heapq
from utils import *

def plot_graph(G: nx.Graph, pos, colors=None, edge_labels=None):
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='black')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def dfs(G:nx.Graph, start:int, pos):
    visited = set()
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(set(G[vertex]) - visited)

            colors = ['red' if node in visited else 'blue' for node in G.nodes()]
            edge_labels = nx.get_edge_attributes(G, 'weight')
            plot_graph(G, pos, colors, edge_labels)

    return visited

def bfs(G:nx.Graph, start:int, pos):
    visited = set()
    queue = [start]

    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(set(G[vertex]) - visited)

            colors = ['red' if node in visited else 'blue' for node in G.nodes()]
            edge_labels = nx.get_edge_attributes(G, 'weight')
            plot_graph(G, pos, colors, edge_labels)

    return visited

def ucs(G: nx.Graph, start: int, pos):
    visited = set()
    queue = [(0, start)]  # Priority queue as a list of tuples (cost, node)
    costs = {start: 0}

    while queue:
        cost, vertex = heapq.heappop(queue)
        if vertex not in visited:
            visited.add(vertex)

            # Update the cost for each neighbor
            for neighbor in G[vertex]:
                next_cost = cost + G[vertex][neighbor].get('weight')
                if neighbor not in costs or next_cost < costs[neighbor]:
                    costs[neighbor] = next_cost
                    heapq.heappush(queue, (next_cost, neighbor))

            colors = ['red' if node in visited else 'blue' for node in G.nodes()]
            edge_labels = nx.get_edge_attributes(G, 'weight')
            plot_graph(G, pos, colors, edge_labels)

    return visited, costs


def greedy_search(G: nx.Graph, start: int, goal: int, pos, heuristic):
    visited = set()
    queue = [(0, start)]

    while queue:
        _, vertex = heapq.heappop(queue)
        if vertex == goal:
            break

        if vertex not in visited:
            visited.add(vertex)

            for neighbor in G[vertex]:
                if neighbor not in visited:
                    heuristic_cost = heuristic(pos, neighbor, goal)
                    heapq.heappush(queue, (heuristic_cost, neighbor))

            colors = ['green' if node == goal else 'red' if node in visited else 'blue' for node in G.nodes()]
            edge_labels = nx.get_edge_attributes(G, 'weight')
            plot_graph(G, pos, colors, edge_labels)

    return visited


def a_star_search(G: nx.Graph, start: int, goal: int, pos, heuristic):
    visited = set()
    queue = [(0, start)]
    actual_costs = {start: 0}

    while queue:
        _, vertex = heapq.heappop(queue)
        if vertex == goal:
            break

        if vertex not in visited:
            visited.add(vertex)

            for neighbor in G[vertex]:
                if neighbor not in visited:
                    new_cost = actual_costs[vertex] + G[vertex][neighbor]['weight']
                    if neighbor not in actual_costs or new_cost < actual_costs[neighbor]:
                        actual_costs[neighbor] = new_cost
                        total_cost = new_cost + heuristic(pos, neighbor, goal)
                        heapq.heappush(queue, (total_cost, neighbor))

            colors = ['green' if node == goal else 'red' if node in visited else 'blue' for node in G.nodes()]
            edge_labels = nx.get_edge_attributes(G, 'weight')
            plot_graph(G, pos, colors, edge_labels)

    return visited