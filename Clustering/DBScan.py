import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import squareform
# import morphio
import tmd
import os
import matplotlib.pyplot as plt 

# --- 1. Define Your Custom Distance Function ---
def custom_neuron_distance(neuron1, neuron2):
    # Example custom distance function (replace with your actual implementation)
    pd1 = [tmd.methods.get_persistence_diagram(neurite) for neurite in neuron1.neurites]
    pd2 = [tmd.methods.get_persistence_diagram(neurite) for neurite in neuron2.neurites]

    # Combine persistence diagrams for a full comparison
    combined_pd1 = [item for sublist in pd1 for item in sublist]
    combined_pd2 = [item for sublist in pd2 for item in sublist]

    # Compute the total persistence image difference as the distance
    distance = tmd.Topology.distances.total_persistence_image_diff(combined_pd1, combined_pd2)
    return distance

# --- 2. Load Neuron Data ---
csv_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhNew.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Extract neuron names
neuron_names = data['Name'].values

# Compute pairwise distance matrix
n_neurons = len(neuron_names)
distance_matrix = np.zeros((n_neurons, n_neurons))

for i in range(n_neurons):
    print('.',end=" ")
    for j in range(i + 1, n_neurons):  # Only compute upper triangle (it's symmetric)
        neuro_path1 = f"//global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/{neuron_names[i]}/{neuron_names[i]}/morphology"
        neuro_path2 = f"//global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/{neuron_names[j]}/{neuron_names[j]}/morphology"
        asc_files = [f for f in os.listdir(neuro_path1) if f.endswith('.asc')]

        if not asc_files:
                print(f"No .asc file found for {neuron_names[i]} in {neuro_path1}")
                continue

        morphology_path1 = os.path.join(neuro_path1, asc_files[0])

        asc_files = [f for f in os.listdir(neuro_path2) if f.endswith('.asc')]

        if not asc_files:
                print(f"No .asc file found for {neuron_names[j]} in {neuro_path2}")
                continue

        morphology_path2 = os.path.join(neuro_path2, asc_files[0])

        neuron1 =tmd.io.load_neuron_from_morphio(morphology_path1)
        neuron2 =tmd.io.load_neuron_from_morphio(morphology_path2)
        dist = custom_neuron_distance(neuron1, neuron2)
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist  # Symmetric matrix

# --- 3. Perform DBSCAN Clustering ---
# Convert the distance matrix to a condensed form
print(distance_matrix)
condensed_distance_matrix = squareform(distance_matrix)
condensed_matrix_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/condensed_distance_matrix_SingleInh.npy'
np.save(condensed_matrix_file, condensed_distance_matrix)
print(f"Condensed distance matrix saved to {condensed_matrix_file}")
# Define DBSCAN parameters
eps = 200  # Maximum distance between two samples for them to be considered as in the same neighborhood
min_samples = 2  # Minimum number of samples in a neighborhood to form a core point

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
labels = dbscan.fit_predict(distance_matrix)

# --- 4. Output Results ---
# Print cluster labels
for neuron, label in zip(neuron_names, labels):
    print(f"Neuron: {neuron}, Cluster: {label}")

# Save results to a CSV file
# Save results to a CSV file
output_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/DBScanResultsInhSingle.csv'
results = pd.DataFrame({'Neuron': neuron_names, 'Cluster': labels})
results.to_csv(output_file, index=False)
print(f"DBSCAN clustering results saved to {output_file}")

import matplotlib.pyplot as plt


# --- 5. Plot the Results ---
# Assign colors to clusters
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# Create a scatter plot for visualization
plt.figure(figsize=(10, 7))
y_position = 5  # Fixed y-axis position for all points
for label, color in zip(unique_labels, colors):
   
    label_name = f'Cluster {label}'
    indices = np.where(labels == label)
    for idx in indices[0]:
        # Add small random displacement to x and y positions
        x_displacement = np.random.uniform(-0.2, 0.2)
        y_displacement = np.random.uniform(-0.2, 0.2)
        plt.scatter(label + x_displacement, y_position + y_displacement, c=[color], label=label_name if idx == indices[0][0] else "")
        plt.text(label + x_displacement+0.5, y_position + y_displacement + 0, neuron_names[idx], fontsize=9, ha='center')

plt.title('DBSCAN Clustering Results')
plt.xlabel('Cluster Label')
plt.ylabel('Fixed Y Position')
plt.yticks([y_position], ['Neurons'])
plt.legend()
plt.legend()
plt.tight_layout()
plt.savefig("DBSCAN_results_all_cellsSingle.png")