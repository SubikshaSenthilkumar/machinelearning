import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the Mall Customers dataset
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values  # Use income and spending score for clustering

# 1. K-Means Clustering
# Perform K-Means clustering to get final cluster assignments
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
final_cluster_centers = kmeans.cluster_centers_
error_rate = kmeans.inertia_  # Inertia as error rate
epochs = kmeans.n_iter_       # Number of iterations (epochs)

# Print results in terminal
print("K-Means Clustering Results:")
print(f"Number of iterations (epochs): {epochs}")
print("Final cluster centers (for each cluster):")
for i, center in enumerate(final_cluster_centers, start=1):
    print(f"Cluster {i} center: {center}")
print(f"Error Rate (Inertia): {error_rate:.2f}")

# 2. Plot Final Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=30, c=labels, cmap='viridis')
plt.scatter(final_cluster_centers[:, 0], final_cluster_centers[:, 1], s=300, c='red', marker='X')
plt.title(f"K-Means Clustering with Error Rate (Inertia: {error_rate:.2f})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
