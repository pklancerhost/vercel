# Code Example 3: Cross-Validation Split Concept
data = [10, 20, 30, 40, 50, 60, 70, 80]
folds = 2
fold_size = len(data) // folds

for i in range(folds):
    test_fold = []
    train_fold = []
    
    start = i * fold_size
    end = start + fold_size
    
    for idx in range(len(data)):
        if idx >= start and idx < end:
            test_fold.append(data[idx])
        else:
            train_fold.append(data[idx])
            
    print("Fold", i + 1)
    print("  Train set:", train_fold)
    print("  Test set: ", test_fold)