import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import squareform

# --- 1. Load the Condensed Distance Matrix ---
condensed_matrix_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/condensed_distance_matrixInh.npy'
condensed_distance_matrix = np.load(condensed_matrix_file)
print(f"Loaded condensed distance matrix from {condensed_matrix_file}")

# Convert the condensed distance matrix back to a square form
distance_matrix = squareform(condensed_distance_matrix)

# --- 2. Perform DBSCAN Clustering ---
# Define DBSCAN parameters
eps = 100  # Maximum distance between two samples for them to be considered as in the same neighborhood
min_samples = 3  # Minimum number of samples in a neighborhood to form a core point

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
labels = dbscan.fit_predict(distance_matrix)

# --- 3. Output Results ---
# Load neuron names (if available)
csv_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhNew.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)
neuron_names = data['Name'].values

# Print cluster labels
for neuron, label in zip(neuron_names, labels):
    print(f"Neuron: {neuron}, Cluster: {label}")

# Save results to a CSV file
output_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/DBScanResultsFromCondensedInh.csv'
results = pd.DataFrame({'Neuron': neuron_names, 'Cluster': labels})
results.to_csv(output_file, index=False)
print(f"DBSCAN clustering results saved to {output_file}")

cluster_groups = results.groupby('Cluster')['Neuron'].apply(list)

# Count neurons per cluster
cluster_counts = cluster_groups.apply(len)
import matplotlib.pyplot as plt
import seaborn as sns

# Create the bar plot
plt.figure(figsize=(10, 6))
bars = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='tab10')
max_count = cluster_counts.max()
plt.ylim(0, int(max_count + max(2, max_count * 0.5)))

# Annotate each bar with the neuron list
for i, (cluster, neurons) in enumerate(cluster_groups.items()):
    neuron_list = '\n'.join(neurons)  # newline for better readability
    plt.text(i, cluster_counts[cluster] + 0.2, neuron_list,
             ha='center', va='bottom', fontsize=8, rotation=90, clip_on=True)


plt.subplots_adjust(top=1) 
plt.title('Neuron Count per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Neurons')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Cluster_counts_Inh00.png",bbox_inches='tight')
