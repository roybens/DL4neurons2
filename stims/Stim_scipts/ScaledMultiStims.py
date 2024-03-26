import pandas as pd

def scale_and_save_csv(input_csv, scale_factors, output_prefix):
    df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/"+input_csv+'.csv')
    
    for factor in scale_factors:
        scaled_df = df * factor
        output_csv = f'{input_csv}_{factor:.2f}x.csv'
        scaled_df.to_csv("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/scaled_stims/"+output_csv, index=False)
        print(f'Saved scaled CSV: {output_csv}')

    

# Example usage
input_csvs = ['5k0chaotic5A','5k0step_200','5k0ramp','5k0chirp','5k0chirp','5k50kInterChaoticB','5k0chaotic5B','5k0step_500']  # Replace with the path to your original CSV
scale_factors = [0.25, 0.5, 0.75,1.0,4/3,2.0,4.0]
output_prefix = 'scaled_data'
for input_csv in input_csvs:
    scale_and_save_csv(input_csv, scale_factors, output_prefix)
