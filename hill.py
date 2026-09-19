def hill_climb(state, get_nbrs, value):
    current = state

    while True:
        nbrs = get_nbrs(current)

        best = max(nbrs, key=value, default=None)

        if best is None:
            return current

        if value(best) <= value(current):
            return current

        current = best


# -----------------------------
# Neighbor Function
# -----------------------------
def get_neighbors(x):
    neighbors = []

    if x > 0:
        neighbors.append(x - 1)

    if x < 10:
        neighbors.append(x + 1)

    return neighbors


# -----------------------------
# Value Function
# f(x) = -(x-7)^2 + 49
# Maximum value = 49 at x = 7
# -----------------------------
def value(x):
    return -(x - 7) ** 2 + 49


# -----------------------------
# Starting State
# -----------------------------
start = 2

best_state = hill_climb(start, get_neighbors, value)

print("Starting State :", start)
print("Best State     :", best_state)
print("Maximum Value  :", value(best_state))