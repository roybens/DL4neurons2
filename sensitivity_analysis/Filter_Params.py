import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Step 1: Load the CSV file
file_path = '/global/cfs/projectdirs/m2043/roybens/sens_ana/sen_ana_bbp_Inh/L5_BTC_cAC_0/sensitivityregion_-1_1_L5_BTCcAC.csv'  # Update this with the path to your CSV file
data = pd.read_csv(file_path)

# Set your thresholds here
mean_ecd_threshold = 35  # Example threshold for Mean ECD, adjust as needed
std_ecd_threshold = 2  # Example threshold for STD ECD, adjust as needed

# Step 2: Plotting with thresholds and saving plots into a PDF file
with PdfPages('Filtering_Params_Inh.pdf') as pdf:
    # Plot for Mean ECD
    plt.figure(figsize=(10, 6))
    plt.scatter(data['param_name'], data['Mean ECD'], color='blue', label='Mean ECD')
    plt.axhline(y=mean_ecd_threshold, color='r', linestyle='-', label='Mean ECD Threshold')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Parameter Name')
    plt.ylabel('Mean ECD')
    plt.title('Mean ECD by Parameter')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Plot for STD ECD
    plt.figure(figsize=(10, 6))
    plt.scatter(data['param_name'], data['STD ECD'], color='green', label='STD ECD')
    plt.axhline(y=std_ecd_threshold, color='r', linestyle='-', label='STD ECD Threshold')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Parameter Name')
    plt.ylabel('STD ECD')
    plt.title('STD ECD by Parameter')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

# Step 3: Filter out parameters that do not pass both thresholds
filtered_data = data[(data['Mean ECD'] > mean_ecd_threshold) & (data['STD ECD'] >std_ecd_threshold)]

# Step 4: Save the parameters that pass both thresholds into a text file
passed_parameters = filtered_data['param_name'].values
np.savetxt('passed_parameters.txt', passed_parameters, fmt='%s')

print(f"Filtered parameters saved to 'passed_parameters.txt'. {len(passed_parameters)} parameters passed both thresholds.")
