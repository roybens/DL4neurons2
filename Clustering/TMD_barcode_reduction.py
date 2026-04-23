import numpy as np
import pandas as pd
import tmd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Load Neuron Data ---
# Input CSV should have columns: 'mtype', 'etype', 'i_cell', and optionally 'Name'
input_csv = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/neuron_list.csv'  # Update with your file path
data = pd.read_csv(input_csv)

# Verify required columns
required_columns = ['mtype', 'etype', 'i_cell']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

# Extract neuron information
neuron_info = data[required_columns].copy()
if 'Name' in data.columns:
    neuron_names = data['Name'].values
else:
    # Generate names from mtype_etype_i_cell
    neuron_names = [f"{row['mtype']}_{row['etype']}_{row['i_cell']}" 
                    for _, row in data.iterrows()]
    neuron_info['Name'] = neuron_names

print(f"Processing {len(neuron_names)} neurons...")

# --- 2. Compute TMD Barcodes for Each Neuron ---
def compute_tmd_features(neuron_path):
    """
    Load neuron morphology and compute TMD barcode features.
    Returns a flattened feature vector from persistence diagrams.
    """
    try:
        # Find .asc file in the morphology directory
        asc_files = [f for f in os.listdir(neuron_path) if f.endswith('.asc')]
        
        if not asc_files:
            print(f"No .asc file found in {neuron_path}")
            return None
        
        morphology_path = os.path.join(neuron_path, asc_files[0])
        
        # Load neuron
        neuron = tmd.io.load_neuron_from_morphio(morphology_path)
        
        # Compute persistence diagrams for all neurites
        persistence_diagrams = []
        for neurite in neuron.neurites:
            pd = tmd.methods.get_persistence_diagram(neurite)
            persistence_diagrams.append(pd)
        
        # Convert persistence diagrams to feature vectors
        features = []
        for pd in persistence_diagrams:
            if len(pd) > 0:
                # Extract birth times, death times, and lifetimes
                births = pd[:, 0]
                deaths = pd[:, 1]
                lifetimes = deaths - births
                
                # Compute statistics as features
                features.extend([
                    np.mean(births) if len(births) > 0 else 0,
                    np.std(births) if len(births) > 0 else 0,
                    np.mean(deaths) if len(deaths) > 0 else 0,
                    np.std(deaths) if len(deaths) > 0 else 0,
                    np.mean(lifetimes) if len(lifetimes) > 0 else 0,
                    np.std(lifetimes) if len(lifetimes) > 0 else 0,
                    np.max(lifetimes) if len(lifetimes) > 0 else 0,
                    len(pd)  # Number of topological features
                ])
            else:
                # No persistence diagram for this neurite
                features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Ensure consistent feature vector length (pad with zeros if needed)
        # Assuming max 4 neurites (typical for neurons: apical, basal dendrites, axon)
        max_neurites = 4
        expected_features_per_neurite = 8
        expected_length = max_neurites * expected_features_per_neurite
        
        while len(features) < expected_length:
            features.extend([0] * expected_features_per_neurite)
        
        # Truncate if too long
        features = features[:expected_length]
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error processing {neuron_path}: {str(e)}")
        return None


# Collect all feature vectors
feature_vectors = []
valid_indices = []

for idx, row in neuron_info.iterrows():
    print(f"Processing neuron {idx + 1}/{len(neuron_info)}: {row['Name']}", end=" ")
    
    # Construct path to neuron morphology
    # Adjust this path pattern based on your directory structure
    if 'Name' in data.columns and '/' not in str(row['Name']):
        # Use the Name column if available
        neuron_path = f"/global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/{row['Name']}/{row['Name']}/morphology"
    else:
        # Construct from mtype, etype, i_cell
        # This pattern may need adjustment based on your actual file structure
        neuron_name = f"{row['mtype']}_{row['etype']}_{row['i_cell']}"
        neuron_path = f"/global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/{neuron_name}/{neuron_name}/morphology"
    
    features = compute_tmd_features(neuron_path)
    
    if features is not None:
        feature_vectors.append(features)
        valid_indices.append(idx)
        print("✓")
    else:
        print("✗")

# Convert to numpy array
feature_matrix = np.array(feature_vectors)
print(f"\nSuccessfully processed {len(feature_vectors)} neurons")
print(f"Feature matrix shape: {feature_matrix.shape}")

# --- 3. Dimensionality Reduction to 2D ---
# Standardize features before PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_matrix)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# --- 4. Save Results to CSV ---
# Create results dataframe
results_df = neuron_info.iloc[valid_indices].copy()
results_df['PC1'] = features_2d[:, 0]
results_df['PC2'] = features_2d[:, 1]

output_csv = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/TMD_barcode_2D.csv'
results_df.to_csv(output_csv, index=False)
print(f"\nResults saved to {output_csv}")

# --- 5. Visualize the 2D Projection ---
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)

# Optionally label points with neuron names
for i, name in enumerate(results_df['Name'].values):
    plt.annotate(name, (features_2d[i, 0], features_2d[i, 1]), 
                 fontsize=6, alpha=0.7)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('TMD Barcode Features - 2D PCA Projection')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/TMD_barcode_2D.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {plot_file}")

plt.show()

# --- 6. Optional: Save Full Feature Matrix ---
full_features_csv = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/TMD_full_features.csv'
full_features_df = results_df.copy()
for i in range(feature_matrix.shape[1]):
    full_features_df[f'feature_{i}'] = feature_matrix[:, i]
full_features_df.to_csv(full_features_csv, index=False)
print(f"Full feature matrix saved to {full_features_csv}")
