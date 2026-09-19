# Code Example 2: Confusion Matrix using Standard Python Lists

y_true = ["Sick", "Healthy", "Healthy", "Healthy"]
y_pred = ["Healthy", "Sick", "Healthy", "Healthy"]

# [True Negative, False Positive]
# [False Negative, True Positive]
matrix = [[0, 0], [0, 0]]

for i in range(len(y_true)):
    actual = y_true[i]
    predicted = y_pred[i]

    if actual == "Healthy" and predicted == "Healthy":
        matrix[0][0] = matrix[0][0] + 1  # True Negative

    elif actual == "Healthy" and predicted == "Sick":
        matrix[0][1] = matrix[0][1] + 1  # False Positive

    elif actual == "Sick" and predicted == "Healthy":
        matrix[1][0] = matrix[1][0] + 1  # False Negative

    elif actual == "Sick" and predicted == "Sick":
        matrix[1][1] = matrix[1][1] + 1  # True Positive


# Get values from matrix
TN = matrix[0][0]
FP = matrix[0][1]
FN = matrix[1][0]
TP = matrix[1][1]


# Display Confusion Matrix
print("Confusion Matrix:")
for row in matrix:
    print(row)

print()

# Display explanation
print(f"TN = {TN}: {TN} Healthy person correctly identified as Healthy.")
print(f"FP = {FP}: {FP} Healthy person incorrectly flagged as Sick.")
print(f"FN = {FN}: {FN} Sick person incorrectly identified as Healthy.")
print(f"TP = {TP}: {TP} Sick person correctly identified as Sick.")