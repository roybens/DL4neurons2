import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import pandas as pd

# Load the condensed distance matrix
condensed_matrix_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/condensed_distance_matrix_SingleInh.npy'
condensed_distance_matrix = np.load(condensed_matrix_file)

# Convert the condensed distance matrix back to a square form
distance_matrix = squareform(condensed_distance_matrix)

# Plot the heatmap
# Load neuron names from a CSV file
neuron_names_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhNew.csv'
neuron_names = pd.read_csv(neuron_names_file)['Name'].squeeze().tolist()

plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap='viridis', xticklabels=neuron_names, yticklabels=neuron_names)
plt.title('Heatmap of Distance Matrix')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Save the heatmap
heatmap_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/DistanceMatrixHeatmap.png'
plt.savefig(heatmap_file, bbox_inches='tight')
print(f"Heatmap saved to {heatmap_file}")

# Show the heatmap
plt.savefig("InhHeatmap2.png",bbox_inches='tight')