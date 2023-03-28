import csv
import numpy as np
import os

toExtract = 'distribute($o1.'
for f in os.listdir('./'):
    if '.' in f:
        continue
    saveTo = os.path.join('./', f)
    os.mkdir(saveTo)
    for folder in os.listdir(os.path.join('./', f)):
        if '.' not in folder and folder[-1] == '1':
             filename = os.path.join('./', f, folder + '/biophysics.hoc')
        else:
            continue


        #filename = "./L1/L1_DAC_bNAC219_1/biophysics.hoc"

        params = []
        values = []
        basal = dict()
        apical = dict()
        other = dict()

        # open the file for reading

        filehandle = open(filename, 'r')
        while True:
            # read a single line
            line = filehandle.readline()
            if not line:
                break
            if toExtract in line:
                #param name 
                temp = line[(len(toExtract) + 2):len(line) - 3]
                marker = temp.find(',')
                category = temp[:marker]
                temp = temp[(marker + 2):]
                marker = temp.find("\"")
                paramName = temp[:marker]

                #value
                temp = temp[::-1]
                marker = temp.find('*')
                value = temp[:marker]
                value = round(float(value[::-1]), len(value))

                if category == 'basal':
                    basal[paramName] = value
                elif category == 'apical':
                    apical[paramName] = value
                else:
                    other[paramName + '_' + category] = value

        filehandle.close()

        toRemove = []
        for key in basal.keys():
            if key in apical.keys() and basal[key] == apical[key]:
                other[key + '_' + 'dend'] = basal[key]
                toRemove.append(key)

        for i in toRemove:
            del basal[i]
            del apical[i]

        toSave = [basal, apical, other]
        for i in range(len(toSave)):
            d = toSave[i]
            for k in d.keys(): 
                if i == 0:
                    params.append(k + '_' + 'basal')
                elif i == 1:
                    params.append(k + '_' + 'apical')
                else:
                    params.append(k)
                values.append(d[k])


        params = np.array(params)
        values = np.array(values)
        #lower = values / 10
        #upper = values * 10

        #for i in range(len(values)):
            #lower[i] = round(lower[i], len(str(values[i])) + 1)
            #upper[i] = round(upper[i], len(str(values[i])) + 1)


        zipped = list(zip(params, values))

        with open(os.path.join(saveTo, folder + '_biophysics.csv'), mode='w') as file:
            w = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            w.writerow(['Param name', 'Base value'])
            w.writerows(zipped)
