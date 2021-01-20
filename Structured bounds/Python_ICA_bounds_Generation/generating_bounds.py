#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:09:04 2017

@author: Proteus

"""

""" Calculating ICA bounds"""

# ==== Packages ====

import scipy.io as io
import numpy as np
import ica_tools
from ica_tools import ica_generator

# ==== Loading data ====

label = io.loadmat('label.mat')
label_train = label['yeastlabeltrain']
label_test = label['yeastlabeltest']
label = np.c_[label_train,label_test]

n_train = label_train.shape[1]
n_test = label_test.shape[1]
n_all = n_train + n_test

# ==== Parameters for bounds ====
bound_center = n_all / n_train * np.sum(label_train,axis=1)

# [scale]
#left_width = 0.45 * np.ones([14])
#right_width = 0.45 * np.ones([14])

# [maximum bounds]
width_candidate = np.c_[bound_center, 2417 - bound_center]
width = width_candidate.min(axis = 1)
left_width = width
right_width = width

# [plus]
#small = 20
#big = 100
#margin = bound_center.max()-bound_center.min()
#sigma = (bound_center-bound_center.min())/margin
#width_scale = (1 - sigma) * small + sigma * big
#left_width = width_scale
#right_width = width_scale

bound_left = bound_center - left_width
bound_right = bound_center + right_width
#bound_left = ( 1 - left_width ) * bound_center
#bound_right = ( 1 + right_width) * bound_center

width = [left_width, right_width]
bounds = np.c_[bound_left,bound_right]

# ==== Combination of sources ====

# Graph Laplacian of Classes

Laplacian_class = io.loadmat('L_c.mat')
L_c = Laplacian_class['L_c']

P_temp = []
for i in range(14):
    indicator = L_c[i] != 0
    if indicator.sum() > 0:
        indicator = np.nonzero(indicator)
        indicator = [element for element in indicator[0]]
        P_temp.append(indicator)

# Binary Combination

#P_temp = []
#for i in range(14):
#    for j in range(i+1,14):
#        P_temp.append([i,j])

# ==== Calculating ICA bounds ====

n_P_temp = len(P_temp)
ICA_bounds = []
for i in range(n_P_temp):    
    indicator = P_temp[i]
    n_components = len(indicator)
    label_train_new, bounds_new = ica_tools.data_selector(indicator,
                                                          label_train, bounds)
    ica = ica_generator()
    ica.fit(label_train_new, bounds_new, n_components, n_all)
    ICA_bounds.append(ica.result)
    
# ==== Save file as .mat ====
io.savemat('ICA_bounds', {'ICA_bounds': ICA_bounds,
                          'P_temp': P_temp, 
                          'ICA_width': width,
                          'cubic_bounds': bounds,
                          'center': bound_center[:,np.newaxis]})