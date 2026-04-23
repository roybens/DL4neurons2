import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv('/global/homes/k/ktub1999/mainDL4/DL4neurons2/Clustering/DBScanResultsExcNew.csv') 
# Group neurons by cluster
cluster_groups = df.groupby('Cluster')['Neuron'].apply(list)

# Count neurons per cluster
cluster_counts = cluster_groups.apply(len)

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
plt.savefig("Cluster_counts_Inh.png",bbox_inches='tight')
