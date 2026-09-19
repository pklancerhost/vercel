# import random


# # -----------------------------
# # Fitness Function
# # -----------------------------
# def fitness(x):
#     return x * x


# # -----------------------------
# # Random Population
# # -----------------------------
# population = [random.randint(0, 41) for _ in range(6)]

# print("Initial Population")
# print(population)

# # -----------------------------
# # Run 10 Generations
# # -----------------------------
# for generation in range(10):

#     # Sort by Fitness
#     population.sort(key=fitness, reverse=True)

#     print("\nGeneration", generation + 1)
#     print(population)

#     # Best two become parents
#     parent1 = population[0]
#     parent2 = population[1]

#     # Crossover
#     child = (parent1 + parent2) // 2

#     # Mutation
#     if random.random() < 0.3:
#         child += random.choice([-1, 1])

#     # Keep child within limits
#     child = max(0, min(31, child))

#     # Replace worst chromosome
#     population[-1] = child

# # -----------------------------
# # Best Solution
# # -----------------------------
# population.sort(key=fitness, reverse=True)

# best = population[0]

# print("\nBest Solution =", best)
# print("Maximum Fitness =", fitness(best))
import random

# ---------------------------------
# Step 1: Generate 100 Players
# ---------------------------------

players = []

for i in range(1, 101):
    player = {
        "name": f"Player {i}",
        "fitness": random.randint(40, 100)
    }
    players.append(player)

print("=" * 50)
print("INITIAL 100 PLAYERS")
print("=" * 50)

for player in players:
    print(f"{player['name']:10} Fitness: {player['fitness']}")

# ---------------------------------
# Step 2: Fitness Test
# ---------------------------------

players.sort(key=lambda x: x["fitness"], reverse=True)

# ---------------------------------
# Step 3: Select Top 20
# ---------------------------------

top20 = players[:20]

print("\n")
print("=" * 50)
print("TOP 20 PLAYERS AFTER FITNESS TEST")
print("=" * 50)

for i, player in enumerate(top20, start=1):
    print(f"{i:2}. {player['name']:10} Fitness: {player['fitness']}")

# ---------------------------------
# Step 4: Training & Improvement
# (Mutation)
# ---------------------------------

print("\n")
print("=" * 50)
print("TRAINING SESSION")
print("=" * 50)

for player in top20:

    improvement = random.randint(0, 10)

    old = player["fitness"]

    player["fitness"] = min(100, player["fitness"] + improvement)

    print(f"{player['name']:10} : {old} -> {player['fitness']}")

# ---------------------------------
# Step 5: Select Best Team (11)
# ---------------------------------

top20.sort(key=lambda x: x["fitness"], reverse=True)

best_team = top20[:11]

print("\n")
print("=" * 50)
print("BEST PLAYING XI")
print("=" * 50)

for i, player in enumerate(best_team, start=1):
    print(f"{i:2}. {player['name']:10} Fitness: {player['fitness']}")

print("\n")
print("=" * 50)
print("TEAM AVERAGE FITNESS")
print("=" * 50)

average = sum(player["fitness"] for player in best_team) / len(best_team)

print(f"Average Fitness = {average:.2f}")