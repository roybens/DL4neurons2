"""
Load Jan's predictions, true values, and check closest izhi classification to each
"""
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from models import MODELS_BY_NAME


NSAMPLES = 10000

izhi = MODELS_BY_NAME['izhi']

def normalize(vals, minmax=1):
    mins = np.array([tup[0] for tup in izhi.PARAM_RANGES])
    ranges = np.array([_max - _min for (_min, _max) in izhi.PARAM_RANGES])
    return 2*minmax * ( (vals - mins)/ranges ) - minmax

# Normalized param values to classification name
classes = OrderedDict()
classname_to_params = OrderedDict()

def load_classes():
    with open('modfiles/izhi2003a.mod', 'r') as modfile:
        while modfile.readline().strip() != 'COMMENT':
            pass

        headers = modfile.readline().split()
        modfile.readline() # space between headers and vals

        while True:
            line = modfile.readline().strip()
            if line == "ENDCOMMENT":
                break

            line, class_name = line.split('%')

            line = line.split()
            vals = normalize([float(x) for x in line[:4]])
            
            class_name = class_name.strip('% ')

            classes[tuple(vals)] = class_name
            classname_to_params[class_name] = vals

load_classes()

print('\n'.join(classes.values()))

predictions = np.zeros((NSAMPLES, 4), dtype=np.float64)
with open(sys.argv[1]) as prediction_file:
    for line in prediction_file.readlines():
        recid, vals = line.split('[')

        predictions[int(float(recid.strip()))] = vals.strip('] \n').split()

# true = np.genfromtxt(sys.argv[2])
true = np.zeros((NSAMPLES, 4), dtype=np.float64)
with open(sys.argv[2], 'r') as true_param_file:
    for i, line in enumerate(true_param_file.readlines()):
        true[i] = normalize([float(x) for x in line.split()])


true_class = [-1] * NSAMPLES
predicted_class = [-1] * NSAMPLES
dist_from_truth = [-1] * NSAMPLES
min_truth_dist = [-1] * NSAMPLES
min_pred_dist = [-1] * NSAMPLES
pred_dist = [-1] * NSAMPLES
dist_to_true_class_params = [-1] * NSAMPLES
correct = [-1] * NSAMPLES
dist = lambda x, y: sum((a-b)**2 for a, b in zip(x, y))
for i, (truth, prediction) in enumerate(zip(true, predictions)):
    truth_distances = {dist(truth, class_params): class_name
                       for class_params, class_name in classes.items()}
    prediction_distances = {dist(prediction, class_params): class_name
                            for class_params, class_name in classes.items()}

    dist_from_truth[i] = dist(truth, prediction)
    min_truth_dist[i] = min(truth_distances.keys())
    min_pred_dist[i] = min(prediction_distances.keys())
    
    true_class[i] = truth_distances[min_truth_dist[i]]
    predicted_class[i] = prediction_distances[min_pred_dist[i]]
    correct[i] = (true_class[i] == predicted_class[i])
    dist_to_true_class_params[i] = dist(classname_to_params[true_class[i]], prediction)




print("All")
print(len([x for x in correct if x]), "Correct")
print(len([x for x in correct if not x]), "Wrong")
print()

dist_from_truth = np.array(dist_from_truth)
true_class = np.array(true_class)
min_pred_dist = np.array(min_pred_dist)
dist_to_true_class_params = np.array(dist_to_true_class_params)
correct = np.array(correct)
incorrect = np.logical_not(correct)



# PLOTTING

plt.scatter(min_pred_dist[incorrect], dist_to_true_class_params[incorrect])
plt.xlabel("Distance to predicted class params")
plt.ylabel("Distance to true class params")
plt.show()
plt.clf()

plt.hist(dist_to_true_class_params[incorrect] - min_pred_dist[incorrect], 20)
plt.xlabel("Dist to true class - dist to predicted class")
plt.show()
plt.clf()


def histo(cls, _dist_from_truth, _correct, col=['k', 'r'], bins=None, i=None):
    if cls in classname_to_params:
        cls_i = list(classname_to_params.keys()).index(cls)
        print("C{} ({}): {}/{}".format(cls_i, cls, sum(_correct), len(_correct)))
        
    bins = bins or int(len(_correct)/20)

    if i is not None:
        plt.subplot(4, 5, i+1)
    plt.title(cls)

    if len(_correct) == 0:
        return

    # DEBUG
    bins = 20
    # END DEBUG
    
    plt.hist([_dist_from_truth[_correct], _dist_from_truth[_correct == False]],
             bins, (0, 0.1), color=col, label=['Correct', 'Wrong'])
    plt.axis('equal')
    plt.yscale('log')
    if i is None:
        plt.legend()


histo('All', dist_from_truth, correct, bins=50)
plt.xlabel("Distance to true params")
plt.ylabel("# Guesses")
plt.show()

plt.clf()

for i, cls in enumerate(classes.values()):
    idx = (true_class == cls)
    histo(cls, dist_from_truth[idx], correct[idx], i=i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()
