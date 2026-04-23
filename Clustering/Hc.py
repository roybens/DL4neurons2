import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import tmd
import os
from matplotlib.backends.backend_pdf import PdfPages

# --- 1. Define Your Custom Distance Function ---
def custom_neuron_distance(neuron1, neuron2):
    pd1 = [tmd.methods.get_persistence_diagram(neurite) for neurite in neuron1.neurites]
    pd2 = [tmd.methods.get_persistence_diagram(neurite) for neurite in neuron2.neurites]

    # Combine persistence diagrams for a full comparison
    combined_pd1 = [item for sublist in pd1 for item in sublist]
    combined_pd2 = [item for sublist in pd2 for item in sublist]

    # Compute the total persistence image difference as the distance
    distance = tmd.Topology.distances.total_persistence_image_diff(combined_pd1, combined_pd2)
    return distance

# --- 2. Read Neurons from CSV ---
# Replace 'neurons.csv' with the path to your CSV file
# Ensure the CSV contains the necessary data to represent neurons
neurons_df = pd.read_csv('/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhNew.csv')

# Assuming each row in the CSV represents a neuron and contains its features
# Convert the DataFrame to a NumPy array for processing
neurons = neurons_df['Name']
n_neurons = neurons.shape[0]

print(f"Number of neurons: {n_neurons}")
print("Sample Neuron Data (first 5):\n", neurons[:5])

# --- 3. Compute Pairwise Distance Matrix ---
distance_matrix = np.zeros((n_neurons, n_neurons))

for i in range(n_neurons):
    for j in range(i + 1, n_neurons):  # Only compute upper triangle (it's symmetric)
        neuro_path1 = f"//global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/{neurons[i]}/{neurons[i]}/morphology"
        neuro_path2 = f"//global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/{neurons[j]}/{neurons[j]}/morphology"
        asc_files = [f for f in os.listdir(neuro_path1) if f.endswith('.asc')]

        if not asc_files:
                print(f"No .asc file found for {neurons[i]} in {neuro_path1}")
                continue

        morphology_path1 = os.path.join(neuro_path1, asc_files[0])

        asc_files = [f for f in os.listdir(neuro_path2) if f.endswith('.asc')]

        if not asc_files:
                print(f"No .asc file found for {neurons[j]} in {neuro_path2}")
                continue

        morphology_path2 = os.path.join(neuro_path2, asc_files[0])

        neuron1 =tmd.io.load_neuron_from_morphio(morphology_path1)
        neuron2 =tmd.io.load_neuron_from_morphio(morphology_path2)

        dist = custom_neuron_distance(neuron1, neuron2)
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist  # Symmetric matrix

print("Pairwise Distance Matrix (snippet):\n", distance_matrix[:5, :5])

# --- 4. Perform Hierarchical Clustering ---
condensed_distance_matrix = squareform(distance_matrix, checks=False)

# Perform hierarchical clustering
linked = linkage(condensed_distance_matrix, method='complete')

# Form clusters based on a distance threshold
distance_threshold = 1100
hierarchical_labels = fcluster(linked, t=distance_threshold, criterion='distance')

print(f"Hierarchical Clusters (distance threshold={distance_threshold}):\n", hierarchical_labels)

# --- 5. Visualize the Dendrogram ---
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=neurons.tolist(),
              leaf_rotation=90,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Neuron Index')
plt.ylabel('Distance (based on custom metric)')
with PdfPages("Hc_dendrogram_all.pdf") as pdf:
    plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold ({distance_threshold})')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()  # Save the current figure into the PDF
    plt.close()