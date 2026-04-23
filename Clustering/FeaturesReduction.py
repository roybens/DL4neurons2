import tmd
import os

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt


def custom_neuron_distance(neuron1, neuron2):
    pd1 = [tmd.methods.get_persistence_diagram(neurite) for neurite in neuron1.neurites]
    pd2 = [tmd.methods.get_persistence_diagram(neurite) for neurite in neuron2.neurites]

    # Combine persistence diagrams for a full comparison
    combined_pd1 = [item for sublist in pd1 for item in sublist]
    combined_pd2 = [item for sublist in pd2 for item in sublist]

    # Compute the total persistence image difference as the distance
    # distance = tmd.Topology.distances.total_persistence_image_diff(combined_pd1, combined_pd2)
    # return distance
    max_features = 4
    processed_data = []
    for sample in combined_pd1:
    # Flatten each feature (e.g., [1, 2] -> 1, 2)
        flattened_sample = [value for feature in sample for value in feature]
        
        # Pad with zeros if the sample has fewer features
        if len(flattened_sample) < max_features * 2:  # 2 values per feature
            flattened_sample.extend([0] * (max_features * 2 - len(flattened_sample)))
        
        # Truncate if the sample has more features
        processed_data.append(flattened_sample[:max_features * 2])

    processed_data = np.array(processed_data)

    # Step 2: Apply UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    embedding = reducer.fit_transform(processed_data)

    # Step 3: Visualize the UMAP embedding
    plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(len(data)), cmap='viridis', s=50)
    plt.colorbar(label='Sample Index')
    plt.title('UMAP Projection')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig("UMAP_Projection.png", bbox_inches='tight')




neuro_path1 = f"/global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/L5_TTPC1_cADpyr232_1/L5_TTPC1_cADpyr232_1/morphology/dend-C060114A2_axon-C060114A5.asc"
neuro_path2 = f"/global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate/L6_BPC_cADpyr231_1/L6_BPC_cADpyr231_1/morphology/dend-tkb061006a2_ch0_cl_h_zk_60x_0_axon-tkb060329a1_ch4_cl_o_db_60x_1.asc"
# asc_files = [f for f in os.listdir(neuro_path1) if f.endswith('.asc')]
# morphology_path1 = os.path.join(neuro_path1, asc_files[0])
# morphology_path2 = os.path.join(neuro_path2, asc_files[0])
neuron1 =tmd.io.load_neuron_from_morphio(neuro_path1)
neuron2 =tmd.io.load_neuron_from_morphio(neuro_path2)

dist = custom_neuron_distance(neuron1, neuron2)
