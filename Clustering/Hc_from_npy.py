import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

# Step 1: Load the condensed distance matrix from a .npy file
condensed_distance_matrix_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/condensed_distance_matrixInh.npy'
condensed_distance_matrix = np.load(condensed_distance_matrix_path)

# Step 2: Perform hierarchical clustering
linked = linkage(condensed_distance_matrix, method='complete')

# Step 3: Form clusters based on a distance threshold
distance_threshold = 1200
hierarchical_labels = fcluster(linked, t=distance_threshold, criterion='distance')

# Step 4: Load the CSV file
csv_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhNew.csv'
data = pd.read_csv(csv_path)

# Step 5: Match the Name column to the clusters
if len(data) != len(hierarchical_labels):
    raise ValueError("The number of rows in the CSV does not match the number of clusters.")

data['Cluster'] = hierarchical_labels

# Step 6: Save the results to a new CSV
output_csv_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/Clustered_AllInhNew.csv'
data.to_csv(output_csv_path, index=False)

print(f"Clustered data saved to {output_csv_path}")

# Step 7: Plot a histogram of clusters
plt.figure(figsize=(10, 6))
data['Cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')

# Add labels and title
plt.title('Histogram of Clusters', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.tight_layout()

# Save the histogram as an image
output_image_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/Cluster_HC_Histogram.png'
plt.savefig(output_image_path)

print(f"Histogram saved to {output_image_path}")