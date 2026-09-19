import heapq


def a_star(graph, start, goal, h):
    # (f, g, path)
    frontier = [(h(start), 0, [start])]
    visited = {}

    while frontier:
        f, g, path = heapq.heappop(frontier)
        node = path[-1]

        # Goal reached
        if node == goal:
            return path, g

        # Skip if already visited with lower cost
        if node in visited and visited[node] <= g:
            continue

        visited[node] = g

        # Explore neighbors
        for nxt, cost in graph[node]:
            new_g = g + cost
            new_f = new_g + h(nxt)

            heapq.heappush(frontier, (new_f, new_g, path + [nxt]))

    return None, float("inf")


# -------------------------
# Graph (Weighted)
# -------------------------
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 1)],
    'D': [('G', 5)],
    'E': [('G', 2)],
    'F': [('G', 3)],
    'G': []
}


# -------------------------
# Heuristic Values
# -------------------------
heuristic = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 4,
    'E': 2,
    'F': 1,
    'G': 0
}


def h(node):
    return heuristic[node]


# -------------------------
# Run A*
# -------------------------
path, cost = a_star(graph, 'A', 'G', h)

print("Shortest Path :", path)
print("Total Cost    :", cost)