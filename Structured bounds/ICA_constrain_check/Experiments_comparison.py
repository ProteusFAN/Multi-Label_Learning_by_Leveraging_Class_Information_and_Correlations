#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:00:54 2017

@author: Proteus
"""

# ====Packages====

import scipy.io as io
import numpy as np
import multi_label_comparison
from multi_label_comparison import ica_constrain

# ====Loading data====

label = io.loadmat('label.mat')
data = io.loadmat('yeast_result_with_instance_bound_of_each_class.mat')
data = data['result_struct'][0,0]

# [original label]
label_train = label['yeastlabeltrain']
label_test = label['yeastlabeltest']
n_train = label_train.shape[1]
n_test = label_test.shape[1]
n_all = n_train + n_test

# [True label]
Y = np.c_[label_train,label_test]

# [prediction]
Z = data['Z_01']

# [bounds]
bound_left = data['instance_lower_number']
bound_right = data['instance_upper_number']
#bound_left = 0*np.ones([14,1])
#bound_right = 2417*np.ones([14,1])
bounds = np.c_[bound_left,bound_right]

# [graph laplacian]
class_Laplacian_matrix = data['class_Laplacian_matrix']

# ====All classes====

ica_all = ica_constrain()
ica_all.fit(label_train, 14)
ica_all.transform(np.c_[bound_left,bound_right], n_all)
ica_all.check(Z, 'prediction')
ica_all.check(Y, 'ground_truth')
result_all = ica_all.result

# ====Graph laplacians====
result_laplacian = []

for i in range(14):
    indicator = class_Laplacian_matrix[i] != 0
    n_components = indicator.sum()
    label_train_new, bounds_new, Z_new, Y_new = multi_label_comparison.data_selector(
            indicator, label_train, bounds, Z, Y)
    ica_laplacian = ica_constrain()
    ica_laplacian.fit(label_train_new, n_components)
    ica_laplacian.transform(bounds_new, n_all)
    ica_laplacian.check(Z_new, 'prediction')
    ica_laplacian.check(Y_new, 'ground_truth')
    result_laplacian.append(ica_laplacian.result)

# ====Binary combination====

result_binary = list()

indicator_all = list()
for i in range(14):
    for j in range(i+1,14):
        indicator_all.append([i,j])
        
for indicator in indicator_all:
    label_train_new, bounds_new, Z_new, Y_new = multi_label_comparison.data_selector(
            indicator,label_train,bounds,Z, Y)
    ica_binary = ica_constrain()
    ica_binary.fit(label_train_new, 2)
    ica_binary.transform(bounds_new, n_all)
    ica_binary.check(Z_new, 'prediction')
    ica_binary.check(Y_new, 'ground_truth')
    result_binary.append(ica_binary.result)

TorF = np.array([result_binary[i]['TorF'] for i in range(len(result_binary))])
T_percentage = np.sum(TorF,axis=0)/len(TorF)
    