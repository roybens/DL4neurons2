# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:09:28 2019

@author: bensr
"""
import numpy as np
import logging as log
from neuron import h
prob_loc_fn = './probe-locator-101cells.csv'
probe_dict_101 = {}
def read_101_cell_probe_locator():
    with open(prob_loc_fn) as fp: 
        all_lines = fp.readlines() 
        for line in all_lines:
            line = line.split()
            axprobe = line[3][5:]
            dndprobe = line[4][5:]
            probes = ['soma[0]',f'axon[{axprobe}]',f'dend[{dndprobe}]']
            probe_dict_101[line[2]] = probes
   # print(probe_dict_101)
read_101_cell_probe_locator()

def get_rec_points_101(long_name):
    return(probe_dict_101[long_name])

def create_sampling_map(power2):
    points = np.array([0,1]).tolist()
    for i in range(power2):
        npoints = 2**i
        curr_points = np.linspace(0,1,npoints,False).tolist()
        curr_points = [x for x in curr_points if x not in points]
        points = points + curr_points
        
    return points



def get_sec_list():
    sec_list = []
    sec_dends = []
    sec_axons =[]
    h.load_file('mosinit.hoc')
    h('''
    objref root
    root = new SectionRef()
    if (root.has_parent()){
	root = root.root()
	}
    ''')
    h('objref sec_list,axonal,dendritic')
    h('sec_list = new SectionList()')
    h('axonal = new SectionList()')
    h('dendritic = new SectionList()')
    h('forsec "dend"{dendritic.append()}')
    h('forsec "apic"{dendritic.append()}')
    h('forsec "axon"{axonal.append()}')
    h('access root.sec')
    h('distance()')
    h('sec_list.wholetree()')
    for curr_sec in h.sec_list:
        curr_name = h.secname(sec=curr_sec)
        sec_list.append(curr_name)
    for curr_sec in h.dendritic:
        curr_name = h.secname(sec=curr_sec)
        sec_dends.append(curr_name)
    for curr_sec in h.axonal:
        curr_name = h.secname(sec=curr_sec)
        sec_axons.append(curr_name)
    return sec_list,sec_dends,sec_axons
         

def get_distance(sec_list,dendritic,axonal):
    h.distance()
    distances = []
    axon_dists = []
    dend_dists = []
    for curr_sec in sec_list:
        curr_dist = h.distance(0.5)
        distances.append(curr_dist)
    for curr_sec in dendritic:
        curr_dist = h.distance(0.5)
        dend_dists.append(curr_dist)
    for curr_sec in axonal:
        curr_dist = h.distance(0.5)
        axon_dists.append(curr_dist)
    return distances,axon_dists,dend_dists

def get_recording_points(smap,sec_names,sec_dists):
    sort_inds =np.argsort(sec_dists)
    orderd_dists = np.sort(sec_dists)
    sampled_dists = []
    sampled_names = []
    names = list(sec_names)
    for i in range(min(len(names),len(smap))):
        ind = smap[i]
        dist_ind = int(ind*len(sec_dists)*0.999)
        sampled_dists.append(orderd_dists[dist_ind])
        sampled_names.append(names[sort_inds[dist_ind]])
    return sampled_names,sampled_dists
    
    
def print_secs_dists(sec_names,sec_dists):
    for sec,dist in zip(sec_names,sec_dists):
        print(sec)
        print(dist)
  
def get_rec_points(hobj):
    smap = list(create_sampling_map(6))
    #log.info(str(smap))
    sec_list = hobj.all
    #log.info(str(list(sec_list)))
    sec_dends = hobj.apical
    #log.info(str(sec_dends))
    sec_basal = hobj.basal
    sec_axons = hobj.axonal
    sec_somatic = hobj.somatic
    for curr_sec in sec_basal :
        sec_dends.append(sec=curr_sec)
    sec_dends.unique()
    [distances,axon_dists,dend_dists] = get_distance(sec_list,sec_dends,sec_axons)
    [axon_sampled,axon_sampled_dists] = get_recording_points(smap,sec_axons,axon_dists)
    #log.info(str(axon_sampled))
    [dend_sampled,dend_sampled_dists] = get_recording_points(smap,sec_dends,dend_dists)
    #log.info(str(dend_sampled))
    #log.info(str(len(list(sec_dends))))
    rec_list  = []
    dist_list = []
    for curr_sec in sec_somatic:
        rec_list.append(curr_sec)
        dist_list.append(0)
    if len(axon_sampled)<3:
        for curr_sec,curr_dist in zip(sec_axons,axon_dists):
            rec_list.append(curr_sec)
            dist_list.append(curr_dist)
        for dend_sec,dend_dist in zip(dend_sampled,dend_sampled_dists):
            rec_list.append(dend_sec)
            dist_list.append(dend_dist)
    else:
        for axon_sec,dend_sec,axon_dist,dend_dist in zip(axon_sampled,dend_sampled,axon_sampled_dists,dend_sampled_dists):
            rec_list.append(axon_sec)
            dist_list.append(axon_dist)
            rec_list.append(dend_sec)
            dist_list.append(dend_dist)
    smap=[]
    # print_secs_dists(rec_list,dist_list)
    return rec_list
    
def get_rec_list():
    smap = list(create_sampling_map(6))
    [sec_list,sec_dends,sec_axons] = get_sec_list()
    [distances,axon_dists,dend_dists] = get_distance()
    [axon_sampled,axon_sampled_dists] = get_recording_points(smap,sec_axons,axon_dists)
    [dend_sampled,dend_sampled_dists] = get_recording_points(smap,sec_dends,dend_dists)
    rec_list  = []
    dist_list = []
    for axon_sec,dend_sec,axon_dist,dend_dist in zip(axon_sampled,dend_sampled,axon_sampled_dists,dend_sampled_dists):
        rec_list.append(axon_sec)
        dist_list.append(axon_dist)
        rec_list.append(dend_sec)
        dist_list.append(dend_dist)
    rec_list = [sec_list[0]] + rec_list
    dist_list = [0] + dist_list
    # print_secs_dists(rec_list,dist_list)
    return rec_list


def get_probe_from_dist(rec_dist_list,probe_name,req_distance):
    best_dist = 1e6
    ans_probe = ''
    best_ind = -1
    counter = -1
    dist_from_soma = 1e6
    for pair in rec_dist_list:
        currname = pair[0]
        counter +=1
        if probe_name in currname:
            currdist = pair[1]
            distance_from_req = np.abs(req_distance-currdist)
            if (distance_from_req<best_dist):
                ans_probe = currname
                #print(f'updating: closest probe to {req_distance}  of {probe_name} is {ans_probe} it is {currdist} from soma and {distance_from_req} form desired probe prev {best_dist}')
                best_dist = distance_from_req
                best_ind = counter
                dist_from_soma = currdist
    ans_entry = [best_ind,ans_probe,dist_from_soma]
    return ans_entry


def get_rec_pts_from_distances(hobj,axon_targets = [],dend_targets = []):
    h.distance()
    sec_list = hobj.all
    sec_dends = hobj.apical
    sec_basal = hobj.basal
    sec_axons = hobj.axonal
    sec_somatic = hobj.somatic
    for curr_sec in sec_basal :
        sec_dends.append(sec=curr_sec)
    sec_dends.unique()
    [distances,axon_dists,dend_dists] = get_distance(sec_list,sec_dends,sec_axons)
    rec_list  = []
    dist_list = []
    for curr_sec in sec_somatic:
        rec_list.append(curr_sec)
        dist_list.append(0)
    for curr_dist in axon_targets:
        [idx,probe_name,probe_dist] = get_closest_probe(axon_dists,sec_axons,curr_dist)
        rec_list.append(probe_name)
        dist_list.append(probe_dist)
    
    for curr_dist in dend_targets:
        [idx,probe_name,probe_dist] = get_closest_probe(dend_dists,sec_dends,curr_dist)
        rec_list.append(probe_name)
        dist_list.append(probe_dist)
    print(f'probe names {rec_list} dists {dist_list}')
    return rec_list
