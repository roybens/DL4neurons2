from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# initialize dataframe
n = 200
ngroup = 4
df = pd.DataFrame({'data': np.random.rand(n), 'group': map(np.floor, np.random.rand(n) * ngroup)})

group = 'group'
column = 'data'
grouped = df.groupby(group)



x = []
val=[]




clevels = np.linspace(0., 1.)
plt.boxplot(vals, labels=names)
#ngroup = len(vals)

plt.scatter(x, val, c=cm.prism(clevels), alpha=0.4)

plt.savefig('meantest.png')