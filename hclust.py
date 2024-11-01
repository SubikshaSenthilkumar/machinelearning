import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Mall Customers dataset
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values  # Use income and spending score for clustering

# 1. Initial Clusters - Dendrogram
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("1. Initial Clusters: Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# 2. Final Clusters with Number of Merges (Epochs)
# Perform Agglomerative Clustering to get final cluster assignments
agg_clustering = AgglomerativeClustering(n_clusters=5)
labels = agg_clustering.fit_predict(X)

# Calculate number of merges (epochs) to reach 5 clusters
epochs = len(X) - 5  # Total data points - final clusters
print(f"Number of merges to reach 5 clusters (epochs): {epochs}")

# Plot Final Clusters with Epochs
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=30, c=labels, cmap='viridis')
plt.title(f"2. Final Clusters with Hierarchical Clustering (Merges: {epochs})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# 3. Final Clusters with Error Rate
# Calculate error rate as the sum of squared distances to cluster centers
final_cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(5)])
error_rate = np.sum([np.linalg.norm(X[i] - final_cluster_centers[labels[i]])**2 for i in range(len(X))])

# Print cluster centers and error rate in terminal
print("Final cluster centers (for each cluster):")
for i, center in enumerate(final_cluster_centers, start=1):
    print(f"Cluster {i} center: {center}")
print(f"Error Rate (Inertia): {error_rate:.2f}")

# Plot Final Clusters with Error Rate
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=30, c=labels, cmap='viridis')
plt.scatter(final_cluster_centers[:, 0], final_cluster_centers[:, 1], s=300, c='red', marker='X')
plt.title(f"3. Final Clusters with Error Rate (Inertia: {error_rate:.2f})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
