import logging as log
from neuron import h

from run import get_model
from get_rec_points import get_rec_points
import sys
# model = get_model('BBP', log, 'L4_BTC', 'cNAC', 0)
m_type = sys.argv[1]
e_type = sys.argv[2]
model = get_model('BBP', log, m_type, e_type, 0)

x = get_rec_points(model.entire_cell)
y = get_rec_points(model.entire_cell)

print("First time: {}".format(x))
print("Second time: {}".format(y))

print("First time: {} rec points".format(len(x)))
print("Second time: {} rec points".format(len(y)))
h("segcount=0")
h('forall {segcount+=1}')
print("total number of segments")
h("print segcount")