# Code Example 3: Missing Value Imputation (Replacing Missing Values)
# Using -1 to represent missing data
data_with_missing = [10, 20, -1, 40, -1, 50]

# Calculate total and count for average
total = 0
count = 0

for val in data_with_missing:
    if val != -1:
        total = total + val
        count = count + 1

mean_val = total / count
imputed_data = []

for val in data_with_missing:
    if val == -1:
        imputed_data.append(mean_val)
    else:
        imputed_data.append(val)

print("Imputed Data (Replaced -1 with mean):", imputed_data)