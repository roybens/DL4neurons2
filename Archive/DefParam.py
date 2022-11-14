from run import get_model
import logging as log
import numpy as np
import pandas as pd
ar = []
for i in range(19):
    ar.append([])
def make_paramset_regions(my_model,param_ind,nsamples,nregions):
    def_param_vals = my_model.DEFAULT_PARAMS
    for curr_region in range(nregions):
        curr_lb = -1 + curr_region*(8/nregions)
        curr_ub = -1 + (curr_region+1)*(8/nregions)
        curr_param_set = np.array([def_param_vals]*nsamples)
        curr_Val_lb = def_param_vals[param_ind]* np.exp(curr_lb)
        curr_Val_ub = def_param_vals[param_ind]* np.exp(curr_ub)
        # curr_vals_check=def_param_vals[param_ind]*np.exp(np.random.uniform(curr_lb,curr_ub,size=nsamples)*np.log(10))
        # curr_param_set[:,param_ind] = curr_vals_check
        ar[param_ind].append([curr_Val_lb,curr_Val_ub])
  

# L4_SS_cADpyr
# my_model = get_model('BBP',log,'L4_SS','cADpyr',cell_i=1)
# def_param_vals = my_model.DEFAULT_PARAMS
# print(my_model.PARAM_NAMES)
# print(def_param_vals)
# print("\n")

my_model = get_model('BBP',log,'L4_SS','cADpyr',cell_i=1)
def_param_vals = my_model.DEFAULT_PARAMS
print(my_model.PARAM_NAMES)
print(def_param_vals)

for i in range(19):
    make_paramset_regions(my_model,i,2,4)
print(ar)
df = pd.DataFrame(ar)
df['ParamName']=my_model.PARAM_NAMES
df.columns= ['Region0','Region1','Region2','Region3','ParamName']
df = df[['ParamName','Region0','Region1','Region2','Region3']]
print(df.head(5))
df.to_csv("Param_Ranges.csv")
