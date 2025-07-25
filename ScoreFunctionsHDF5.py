import argparse
import h5py
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import efel
import pandas as pd
from fastdtw import fastdtw

import warnings
warnings.filterwarnings("ignore")

# Placeholder function for computing custom features
# Returns a dictionary where keys are feature names and values are NumPy arrays
def compute_features(voltage_trace):
    # Replace this with the actual feature computation logic
    dt = 0.1

    features_to_compute = [
        'mean_frequency', 'AP_amplitude', 'AHP_depth_abs_slow',
        'fast_AHP_change', 'AHP_slow_time',
        'spike_half_width', 'time_to_first_spike', 'inv_first_ISI', 'ISI_CV',
        'ISI_values','adaptation_index'
    ]

    trace1 = {}
    timsteps = len(voltage_trace)
    trace1['T'] = [x*dt for x in range(len(voltage_trace))]
    trace1['V'] = voltage_trace
    trace1['stim_start'] = [0]
    trace1['stim_end'] = [timsteps*dt]
    
    feature_values_list = efel.get_feature_values([trace1], features_to_compute)
    
    return feature_values_list[0]

def load_volts(path,stim_index=0):
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')])
    
    thread_id = int(os.environ['SLURM_PROCID'])
    total_threads = int(os.environ['SLURM_NPROCS'])
    files_per_thread = len(files)//total_threads
    # files_per_thread = 8
    files = files[thread_id*files_per_thread:(thread_id+1)*files_per_thread]


    volts = []
    for file in files:
        with h5py.File(file, 'r') as hf:
            volts.append(hf['volts'][:,:,0,stim_index],)
    volts = np.array(volts)
    volts = volts.reshape(volts.shape[0]*volts.shape[1],volts.shape[2],-1)
    # volts = volts[:400,:,:]
    return volts

def load_votls_single(path,size):
    all_volts = np.zeros(size)
    with h5py.File(path,'r') as hf:
        volts = np.array(hf['volts'][0,:,0,0])
    all_volts[:,:,0] = volts[:]
    return all_volts

def collect_features(volts):
    feature_dict = {}
    for volts_data in volts:
        features = compute_features(volts_data)
        for feature_name, feature_values in features.items():
            if feature_name not in feature_dict:
                feature_dict[feature_name] = []
            feature_dict[feature_name].append(feature_values)
    
    # Convert lists to NumPy arrays for each feature
    # for feature_name in feature_dict:
    #     feature_dict[feature_name] = np.concatenate(feature_dict[feature_name])
    
    return feature_dict

def safe_mean(lis):
    if np.size(lis) == 0:
        return 0
    return np.mean(lis)

def compute_mse_per_feature(list1,list2):
    if(list1 is None):
        list1 = np.array([0])
    if(list2 is None):
        list2 = np.array([0])
    len1, len2 = len(list1), len(list2)
    if len1 > len2:
        list2 = np.concatenate((list2, np.zeros(len1 - len2)), axis=0)
    if len2> len1:
        list1 = np.concatenate((list1, np.zeros(len2 - len1)), axis=0)

    return np.sqrt(safe_mean((list1 - list2)**2))



def compute_mse(volts_actual, volts_predict):
    # Collect features for both actual and predicted volts
    features_actual = collect_features(volts_actual)
    features_predict = collect_features(volts_predict)
    mse_dict = {}

    # Compute MSE for each feature and apply MinMax scaling
    scaler = MinMaxScaler()

    for feature_name in features_actual:
        mse_arr = []
        actual_values = features_actual[feature_name]
        predict_values = features_predict[feature_name]

        # Compute MSE for this feature
        if(len(actual_values)!=len(predict_values)):
            print("actual and predicted size diff?")
        for actual_value, predict_value in zip(actual_values,predict_values):
            mse_arr.append(compute_mse_per_feature(actual_value,predict_value))
        mse_arr = np.array(mse_arr)
        mse_arr_max = np.nanmax(mse_arr)
        if mse_arr_max == np.inf:
            mse_arr_max =0
        mse_arr[np.where(np.isnan(mse_arr))]=mse_arr_max
        mse_arr[np.isfinite(mse_arr)==False]=mse_arr_max
        mse_arr = scaler.fit_transform(np.array(mse_arr).reshape(-1,1))
        
        # Compute MSE for this feature
        mse_dict[feature_name] = mse_arr.squeeze()

    total_mse_sum = np.zeros(volts_actual.shape[0])
    for features,mse_vals in mse_dict.items():
        total_mse_sum+=mse_vals
    
    

    return mse_dict, total_mse_sum

#Compute DynamicTime Warp Distance 
def compute_DTWD(volts_actual, volts_predict):
    dtw_distances = []
    for i in range(volts_actual.shape[0]):
        x = volts_actual[i].squeeze()
        y = volts_predict[i].squeeze()
        distance, path = fastdtw(x, y)
        dtw_distances.append(distance)
    return dtw_distances

def plot_violin(mse_values):
    sns.violinplot(data=mse_values)
    plt.title("Violin plot of MSE distribution")
    plt.xlabel("Features")
    plt.ylabel("MSE (normalized)")
    plt.savefig("")

def save_to_csv(csv_save_path, total_mse_sum,dynamic_time_warp_distance,csv_name):
    df = pd.DataFrame.from_dict({'Score_Error':total_mse_sum,
                                 'DTWD_Error':dynamic_time_warp_distance   
                                    })
    file_path = os.path.join(csv_save_path,csv_name)

    if not os.path.exists(file_path):
        os.mkdir(file_path)
    csv_path = os.path.join(file_path,str(os.environ['SLURM_PROCID']))
    df.to_csv(csv_path+'MSEScoreError.csv')


def main(path1, path2, csv_name, stim_index):
    
    volts_predict = load_volts(path1,stim_index)
    volts_actual = load_volts(path2,stim_index)
    # volts_predict = load_votls_single(path2, volts_actual.shape)

    mse_values, total_mse_sum = compute_mse(volts_actual, volts_predict)
    dynamic_time_warp_distance = compute_DTWD(volts_actual, volts_predict)
    print(f"Total MSE sum (normalized): {total_mse_sum}")
    save_to_csv(path1,total_mse_sum,dynamic_time_warp_distance,csv_name)
    # plot_violin(total_mse_sum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process two directories of HDF5 files.')
    parser.add_argument('--path1', type=str,required=False,default ="/pscratch/sd/k/ktub1999/BBP_TEST1/runs2/29389166_1/L5_TTPC1_cADpyr232_1/c1" , help='Path to the first directory containing HDF5 files (actual volts)')
    parser.add_argument('--path2', type=str,required=False,default ="/pscratch/sd/k/ktub1999/BBP_TEST1/runs2/29389166_1/L5_TTPC1_cADpyr232_1/c1", help='Path to the second directory containing HDF5 files (predicted volts)')
    parser.add_argument('--csv_name', type=str, required=False, default="MSE_Score_PlotsResults", help='Name of the CSV file to save results')
    parser.add_argument('--stim_index', type=int, required=False, default=0, help='Index of the stimulus to use for feature extraction')
    
    args = parser.parse_args()
    
    main(args.path1, args.path2,args.csv_name, args.stim_index)
# srun --ntasks 128 shifter python3 -m pdb ScoreFunctionsHDF5.py --path1 /pscratch/sd/k/ktub1999/BBP_TEST1/runs2/29645430_1/L5_TTPC1cADpyr0 --path2 /pscratch/sd/k/ktub1999/BBP_TEST1/runs2/29645431_1/L5_TTPC1cADpyr0
# srun --ntasks 128 shifter python3 ScoreFunctionsHDF5.py --path1 /pscratch/sd/k/ktub1999/BBP_TEST1/runs2/30562775_1/L5_BTCcAC0 --path2 /pscratch/sd/k/ktub1999/BBP_TEST1/runs2/30562776_1/L5_BTCcAC0
#salloc -N1 -t 4:00:00 -q interactive -C cpu --image=balewski/ubu20-neuron8:v5