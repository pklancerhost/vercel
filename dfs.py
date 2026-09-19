def dfs(graph, start, goal, path=None):
    path = path or [start]

    if start == goal:
        return path

    for nxt in graph[start]:
        if nxt not in path:
            res = dfs(graph, nxt, goal, path + [nxt])
            if res:
                return res

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

# Run DFS
result = dfs(graph, "A", "F")

print("DFS Path:", result)