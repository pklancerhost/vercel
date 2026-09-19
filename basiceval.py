# Code Example 1: Basic Evaluation Metrics (Accuracy, Precision, Recall)
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

correct = 0
true_positives = 0
false_positives = 0
false_negatives = 0

for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
        correct = correct + 1
    
    if y_true[i] == 1 and y_pred[i] == 1:
        true_positives = true_positives + 1
    elif y_true[i] == 0 and y_pred[i] == 1:
        false_positives = false_positives + 1
    elif y_true[i] == 1 and y_pred[i] == 0:
        false_negatives = false_negatives + 1

accuracy = correct / len(y_true)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)