import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

# Define the directory containing the CSV files
directory = "/global/cfs/projectdirs/m2043/roybens/sens_ana/sens_ana_inh_may6/L4_BTC_bIR_0/"

# Initialize dictionaries to store data for each category
dfs = {'soma': [], 'axon': [], 'dend': [], 'api': []}

exclude = [
    'gK_Tstbar_K_Tst_axonal', 'gNap_Et2bar_Nap_Et2_axonal', 'gImbar_Im_axonal',
    'gNap_Et2bar_Nap_Et2_somatic', 'gK_Pstbar_K_Pst_somatic', 'gImbar_Im_somatic',
    'gCabar_Ca_somatic', 'gSKv3_1bar_SKv3_1_dend', 'gNap_Et2bar_Nap_Et2_dend',
    'gImbar_Im_dend', 'gkbar_StochKv_somatic', 'gkbar_KdShu2007_somatic',
    'gkbar_StochKv_dend', 'gkbar_KdShu2007_dend'
]
exclude = 'gK.Tstbar.K.Tst.axn, gNap.Et2bar.Nap.Et2.axn, gImbar.Im.axn, gNap.Et2bar.Nap.Et2.som, gK.Pstbar.K.Pst.som, gImbar.Im.som, gCabar.Ca.som, gSKv3.1bar.SKv3.1.den, gNap.Et2bar.Nap.Et2.den, gImbar.Im.den, gkbar.StochKv.som, gkbar.KdShu2007.som, gkbar.StochKv.den, gkbar.KdShu2007.den'
exclude = exclude.split(', ')
exclude=[]
# Loop over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        
        # Append the dataframe to the appropriate list based on the filename suffix
        if 'soma' in filename:
            dfs['soma'].append(df)
        elif 'axon' in filename:
            dfs['axon'].append(df)
        elif 'dend' in filename:
            dfs['dend'].append(df)
        elif 'api' in filename:
            dfs['api'].append(df)

# Function to compute Mean and STD of ECD for a list of dataframes, storing results directly
# def compute_means(dataframes):
#     for i, df in enumerate(dataframes):
#         dataframes[i] = df.describe().loc[['mean', 'std']]

# # Apply the function to calculate mean and STD directly within each list of dataframes
# for key in dfs.keys():
#     compute_means(dfs[key])

# Assuming each list has at least one DataFrame and each DataFrame has 'mean STD' calculated
weighted_sum = dfs['soma'][0]['Mean ECD'] + 0.7 * dfs['axon'][0]['Mean ECD'] + 0.2 * dfs['dend'][0]['Mean ECD'] + 0.2 * dfs['api'][0]['Mean ECD']
weighted_sum_std = dfs['soma'][0]['STD ECD'] + 0.7 * dfs['axon'][0]['STD ECD'] + 0.2 * dfs['dend'][0]['STD ECD'] + 0.2 * dfs['api'][0]['STD ECD']
weighted_sum = dfs['soma'][0]['Mean ECD'] +  dfs['axon'][0]['Mean ECD'] +  dfs['dend'][0]['Mean ECD'] +  dfs['api'][0]['Mean ECD']
weighted_sum_std = dfs['soma'][0]['STD ECD'] + dfs['axon'][0]['STD ECD'] +  dfs['dend'][0]['STD ECD'] +  dfs['api'][0]['STD ECD']
weighted_sum = np.clip(weighted_sum,0,500)
weighted_sum_std = np.clip(weighted_sum,0,500)

# Save plots to a PDF
with PdfPages('All_probes_Inh_L4_BTC_bIR_0.pdf') as pdf:
    plt.figure(figsize=(10, 6))
    # plt.scatter(dfs['soma'][0]['param_name'],weighted_sum)
    for i, param in enumerate(dfs['soma'][0]['param_name']):
        color = 'red' if param in exclude else 'blue'
        plt.scatter(i, weighted_sum[i], color=color)
    # weighted_sum.plot(kind='bar')
    plt.title('Weighted Sum of Mean ECD Parameters')
    plt.xlabel('Parameters')
    plt.ylabel('Weighted Mean ECD Value')
    # plt.grid(True)
    plt.xticks(rotation=90)
    plt.xticks(range(len(dfs['soma'][0]['param_name'])), dfs['soma'][0]['param_name'], rotation=90)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    #STD plots
    plt.figure(figsize=(10, 6),layout="tight")
    # weighted_sum.plot(kind='bar')
    for i, param in enumerate(dfs['soma'][0]['param_name']):
        color = 'red' if param in exclude else 'blue'
        plt.scatter(i, weighted_sum_std[i], color=color)
    plt.title('Weighted Sum of STD ECD Parameters')
    plt.xlabel('Parameters')
    plt.ylabel('Weighted STD ECD Value')
    plt.xticks(rotation=90)
    plt.xticks(range(len(dfs['soma'][0]['param_name'])), dfs['soma'][0]['param_name'], rotation=90)
    # plt.grid(True)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    x_positions = range(len(dfs['soma'][0]['param_name']))
    
    plt.figure(figsize=(10, 6))
    plt.scatter([x - 0.2 for x in x_positions], np.clip(dfs['soma'][0]['Mean ECD'],0,500), label='Soma Mean')
    plt.scatter([x - 0.2 for x in x_positions], np.clip(dfs['axon'][0]['Mean ECD'],0,500), label='Axon Mean')
    plt.scatter([x - 0.2 for x in x_positions], np.clip(dfs['dend'][0]['Mean ECD'],0,500), label='dend Mean')
    plt.scatter([x - 0.2 for x in x_positions], np.clip(dfs['api'][0]['Mean ECD'],0,500), label='apic Mean')
    plt.xticks(range(len(dfs['soma'][0]['param_name'])), dfs['soma'][0]['param_name'], rotation=90)
    plt.title('Comparison of Mean Values')
    plt.xlabel('Parameters')
    plt.ylabel('Mean Values')
    plt.legend()
    plt.xticks(rotation=90)
    # plt.grid(True)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

