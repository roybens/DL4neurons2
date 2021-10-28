import csv
import numpy as np
import os
import shutil


readFrom =  "E:/GitHub/DL4neurons/hoc_templates/hoc_combos_syn.1_0_10.allzips/"
saveTo = './'
folders = ["L1/", "L23/", "L4/", "L5/", "L6/"]

def create_etype_nparam_dict():
    with open('etypes_params_list.csv') as csvfile:
        etype_reader = csv.reader(csvfile, delimiter=',')
        etypes_list = next(etype_reader)
        print (etypes_list)
        param_num = np.zeros(len(etypes_list))
        for row in etype_reader:
            tmp =[int(s is not '') for s in row]
            param_num = param_num+tmp
    res = {etypes_list[i]: param_num[i] for i in range(len(etypes_list))} 
    return res
    

count = 1
shortName = []
longName = []
types = []
variedParamsNum = []
all_etypes=[]
def iterate_over_tree():
    count =0
    etype_dict = create_etype_nparam_dict()
    etypes =  list(etype_dict.keys())
    for fn in os.listdir(readFrom):
        if fn is '.' or "_1" not in fn:
            continue
        shortName.append("bbp" + ('0' * (4-len(str(count)))) + str(count))
        count += 1
        longName.append(fn[:-4])
        tmp = [substr in fn for substr in etypes]
        print(fn)
        curr_etype = [x for x in etypes if x in fn][0]
        print(curr_etype)
        
        all_etypes.append(curr_etype)
        variedParamsNum.append(etype_dict[curr_etype])
    zipped = list(zip(shortName, longName, all_etypes, variedParamsNum))
    with open(os.path.join(saveTo, 'extracted_etypes.csv'), mode='w', newline='') as file:
        w = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, )
        w.writerow(["Short Name", "Long Name", "etype", "Number of Varied Params"])
        w.writerows(zipped)
        
iterate_over_tree()