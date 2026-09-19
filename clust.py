# Code Example 3: K-Means Clustering
from sklearn.cluster import KMeans

# Feature points on a grid
X_data = [[1, 2], [1, 4], [10, 2], [10, 4]]

# Group into 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_data)

labels = kmeans.labels_

for i in range(len(X_data)):
    print("Point", X_data[i], "is in Cluster:", labels[i])