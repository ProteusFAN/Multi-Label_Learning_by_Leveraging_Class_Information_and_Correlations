#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:22:32 2017

@author: Proteus
"""

import numpy as np
from sklearn.decomposition import FastICA

class ica_generator(object):
    def __init__(self):
        
        self.result = {'n_components': None, 'n_instance': None, 'mean': None,
                       'un_mixing': None, 'bounds_': None}
        
    def fit(self, Y, bounds, n_components, n_instance):
        
        # ICA fiting process
        self.Y = Y # Y is composed by instance as column.
        self.n_components = n_components # the number of sources
        
        ica = FastICA( n_components = n_components )
        ica.fit(Y.T)
        self.A_ = ica.mixing_
        self.A_inv = ica.components_
        self.mean_ = ica.mean_[:,np.newaxis]
        
        # ICA bounds calculation
        self.bounds = bounds
        self.n_instance = n_instance # the number of all instances
        self.center = np.sum( bounds, axis=1 )/2
        self.center_ = np.dot(self.A_inv,
                              self.center[:,np.newaxis] - self.mean_*n_instance)
        self.vertice = ica_generator.__binary_generator( bounds )
        self.vertice_ = np.dot(self.A_inv,
                               self.vertice - self.mean_*n_instance)
        self.bounds_ = np.zeros([n_components, 2])
        for n_iter in range(n_components):
            candidates = self.vertice_[n_iter]
            left = candidates[candidates <= self.center_[n_iter]]
            right = candidates[candidates >= self.center_[n_iter]]
            self.bounds_[n_iter,0] = left.max()
            self.bounds_[n_iter,1] = right.min()
            
        # Recoding into result
        self.result['n_components'] = self.n_components
        self.result['n_instance'] = self.n_instance
        self.result['mean'] = self.n_instance * self.mean_
        self.result['un_mixing'] = self.A_inv
        self.result['bounds_'] = self.bounds_
        
        
    def __binary_generator(bounds):
        
        if bounds.size == 2:
            return np.array([bounds[0,0],bounds[0,1]])[np.newaxis,:]
        else:
            last = ica_generator.__binary_generator(bounds[1:,:])
            temp_down = np.c_[last, last]
            num_last = last.shape[1]
            temp_up_0 = bounds[0,0]*np.ones([1,num_last])
            temp_up_1 = bounds[0,1]*np.ones([1,num_last])
            temp_up = np.c_[temp_up_0, temp_up_1]
            return np.r_[temp_up, temp_down]
    
def data_selector(indicator, *collection):
    
    # indicator could be a boolean or integer(cumulative) vector.
    result = []
    for element in collection:
        result.append(element[indicator])
    return result