from collections import deque

def bfs(graph, start, goal):
    frontier = deque([[start]])
    visited = {start}

    while frontier:
        path = frontier.popleft()
        node = path[-1]

        if node == goal:
            return path

        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                frontier.append(path + [nxt])

    return None


# Graph
graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["F"],
    "F": []
}

# Call BFS
result = bfs(graph, "A", "F")

# Print result
print("Shortest Path:", result)