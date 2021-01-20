#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:00:54 2017

@author: Proteus
"""

# ====Packages====

import scipy.io as io
import numpy as np
import multi_label
from multi_label import ica_constrain

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
bounds_all_ = ica_all.transform(np.c_[bound_left,bound_right], n_all)
result_all = ica_all.check(Z)
#result_all = ica_all.check(ica_constrain.center[:,np.newaxis])
#result_part = ica_all.check_part(Z,2)

# ====Graph laplacians====

result_laplacian = list()

for i in range(14):
    indicator = class_Laplacian_matrix[i] != 0
    n_components = indicator.sum()
    label_train_new, bounds_new, Z_new = multi_label.data_selector(
            indicator, label_train, bounds, Z)
    ica_laplacian = ica_constrain()
    ica_laplacian.fit(label_train_new, n_components)
    bounds_laplacian_ = ica_laplacian.transform(bounds_new, n_all)
    temp_result = ica_laplacian.check(Z_new)
    result_temp = [indicator, temp_result, ica_laplacian.constrain_indicator,
                   ica_laplacian.Z_sum_, ica_laplacian.bounds_, ica_laplacian]
    result_laplacian.append(result_temp)

# ====Binary combination====

result_binary = list()

indicator_all = list()
for i in range(14):
    for j in range(i+1,14):
        indicator_all.append([i,j])
        
for indicator in indicator_all:
    label_train_new, bounds_new, Z_new = multi_label.data_selector(
            indicator, label_train, bounds, Z)
    ica_binary = ica_constrain()
    ica_binary.fit(label_train_new, 2)
    bounds_binary_ = ica_binary.transform(bounds_new, n_all)
    temp_result = ica_binary.check(Z_new)
    result_temp = [indicator, temp_result, ica_binary.constrain_indicator,
                   ica_binary.Z_sum_, ica_binary.bounds_, ica_binary]
    result_binary.append(result_temp)

n_non_violation = 0
for element in result_binary:
    n_non_violation += element[1]
percent_violation = 1 - n_non_violation/len(result_binary)
    
    