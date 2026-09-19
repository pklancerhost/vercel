# Code Example 2: K-Nearest Neighbors Classifier (k-NN)
from sklearn.neighbors import KNeighborsClassifier

# Features: [Feature 1, Feature 2]
X_train = [[1, 1], [1, 2], [5, 5], [6, 5]]
y_train = [0, 0, 1, 1]  # Classes 0 and 1

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

new_point = [[5, 4]]
prediction = knn.predict(new_point)

if prediction[0] == 0:
    print("k-NN Prediction: Class 0")
else:
    print("k-NN Prediction: Class 1")