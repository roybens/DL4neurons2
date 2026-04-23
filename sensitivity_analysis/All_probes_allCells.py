import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.backends.backend_pdf
import numpy as np


def get_parameter_labels(df):
    """Return parameter labels from common column names, else fallback to index."""
    candidate_cols = [
        'param_name', 'param.name',
        'Parameter', 'parameter', 'Parameters', 'parameters',
        'Parameter Name', 'parameter_name', 'param', 'Param'
    ]

    for col in candidate_cols:
        if col in df.columns:
            return df[col].astype(str).tolist()

    # Fallback: use first non-ECD text-like column if available
    excluded_cols = {'Mean ECD', 'STD ECD'}
    for col in df.columns:
        if col in excluded_cols:
            continue
        if df[col].dtype == object:
            return df[col].astype(str).tolist()

    # If no explicit parameter-name column exists, use row indices as labels
    if 'Mean ECD' in df.columns:
        return [f'P{i + 1}' for i in range(len(df['Mean ECD']))]

    return [f'P{i + 1}' for i in range(len(df))]

def read_and_plot_etype_data(base_folder, csv_path, group_by='etype'):
    """
    Process and plot data based on etypes and their corresponding folders and CSVs.
    
    Parameters:
        base_folder (str): The path to the folder containing cell-specific subfolders.
        csv_path (str): The path to the CSV file containing 'Modified Cell', 'mtype', and 'etype'.
        group_by (str): Grouping/color choice. Either 'etype' or 'mtype'.
    """
    metadata = pd.read_csv(csv_path)

    # Allow common typo as alias
    if group_by == 'etpe':
        group_by = 'etype'

    if group_by not in ['etype', 'mtype']:
        raise ValueError("group_by must be either 'etype' or 'mtype'")

    required_cols = ['Modified Cell', 'mtype', 'etype']
    missing_cols = [c for c in required_cols if c not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s) in metadata CSV: {missing_cols}")

    grouped = metadata.groupby(group_by)

    # Collect data per group first, then plot all groups together per category
    categories = ['soma', 'axon', 'dend', 'api']
    grouped_data = {}
    grouped_labels = {}

    for group_name, group_df in grouped:
        group_folders = []
        for _, row in group_df.iterrows():
            mtype = str(row['mtype'])
            etype = str(row['etype'])

            # Folder format example: L5_TTPC1_cADpyr_0
            folder_pattern = os.path.join(base_folder, f'{mtype}_{etype}_*')
            matches = glob.glob(folder_pattern)

            # Fallback: permissive match if naming has extra tokens
            if not matches:
                fallback_pattern = os.path.join(base_folder, f'*{mtype}*{etype}*')
                matches = glob.glob(fallback_pattern)

            group_folders.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        group_folders = [f for f in group_folders if not (f in seen or seen.add(f))]

        data = {'soma': [], 'axon': [], 'dend': [], 'api': []}
        parameter_labels = {'soma': None, 'axon': None, 'dend': None, 'api': None}

        for folder in group_folders:
            csv_files = glob.glob(os.path.join(folder, '*.csv'))
            for file_path in csv_files:
                category = None
                if 'soma' in file_path:
                    category = 'soma'
                elif 'axon' in file_path:
                    category = 'axon'
                elif 'dend' in file_path:
                    category = 'dend'
                elif 'api' in file_path:
                    category = 'api'

                if category:
                    csv_df = pd.read_csv(file_path)
                    if 'Mean ECD' not in csv_df.columns:
                        continue

                    mean_ecd = np.clip(csv_df['Mean ECD'].values, 0, 250)
                    data[category].append(mean_ecd)

                    if parameter_labels[category] is None:
                        parameter_labels[category] = get_parameter_labels(csv_df)

        grouped_data[group_name] = data
        grouped_labels[group_name] = parameter_labels

    group_names = list(grouped_data.keys())

    # One page per category; all groups (mtypes/etypes) shown together with different colors
    for cat in categories:
        fig, ax = plt.subplots(figsize=(16, 8))
        plotted_any = False
        default_labels = None

        for gi, group_name in enumerate(group_names):
            data = grouped_data[group_name][cat]
            if not data:
                continue

            combined_data = pd.DataFrame(data).T
            x = np.arange(combined_data.shape[0])
            color = colors[gi % len(colors)]

            # slight x offset by group to improve visibility
            offset = (gi - (len(group_names) - 1) / 2.0) * 0.02
            for col in combined_data.columns:
                ax.plot(x + offset, combined_data[col].values, '.', color=color, alpha=0.7)

            if default_labels is None:
                labels = grouped_labels[group_name][cat]
                if labels is not None and len(labels) == len(x):
                    default_labels = labels
                else:
                    default_labels = [f'P{i + 1}' for i in x]

            # add legend handle once per group
            ax.plot([], [], '.', color=color, label=str(group_name))
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xticks(np.arange(len(default_labels)))
        ax.set_xticklabels(default_labels, rotation=90, fontsize=8)
        ax.set_title(f'All {group_by} grouped together - {cat.capitalize()}')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Mean ECD')
        ax.legend(title=group_by, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# Usage
colors = ['b','r','g','y','c','m','orange','k','orchid','peru']
base_folder_path = '//global/cfs/cdirs/m2043/roybens/sens_ana/Feb26L5TTPCMean/'
csv_file_path = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv'

# Choose grouping/color coding: 'mtype' or 'etype' (also accepts typo 'etpe')
group_by_choice = 'mtype'

pdf = matplotlib.backends.backend_pdf.PdfPages("ExcMeanBase.pdf")
read_and_plot_etype_data(base_folder_path, csv_file_path, group_by=group_by_choice)
pdf.close()
