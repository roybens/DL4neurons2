import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Define the folder containing the CSV files
folder_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/KevinStimsFeb2024/Exp_Stims_Long'

# Define the PDF file where the plots will be saved
pdf_filename = 'plots_all.pdf'

# Initialize a PDF pages object
with PdfPages(pdf_filename) as pdf:
    # Iterate over each file in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            with open(file_path, mode='r', newline='') as file:
                # csv_reader = csv.reader(file)
                data = list(csv.reader(file, delimiter=","))
                file.close()
                data = [float(row[0]) for row in data]
                
                # Convert the CSV data into lists, separating headers and values
                
                  # Transpose to get columns
                
                # Plotting
                plt.figure()
                x = np.linspace(0,len(data)+1,len(data),endpoint=False)
                plt.plot(x,data,'r')

                plt.title(filename)
                plt.xlabel('pA')
                plt.ylabel('timebins')
                plt.legend()
                plt.tight_layout()
                
                # Save the current plot to the PDF
                pdf.savefig()
                plt.close()

print(f"All plots have been saved to {pdf_filename}.")
