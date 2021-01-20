#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:42:43 2017

@author: Proteus
"""
import numpy as np
from sklearn.decomposition import FastICA

class ica_constrain(object):
    def __init__(self):
        self.result = {'bounds': None, 'bounds_':  None, 'name': [],
                       'TorF': [], 'constrain_indicator': [],
                       'Z_sum': [], 'Z_sum_': [],
                       'diff_left': [], 'diff_right': []}
        self.result_part = {'bounds': None, 'bounds_': None, 'name': [],
                            'TorF': [], 'constrain_indicator': [],
                            'Z_sum': [], 'Z_sum_': [],
                            'diff_left': [], 'diff_right': []}
        self.check_counter = 0
        self.check_counter_part = 0
        
    def fit(self, Y, n_components, random = True):
        # Y is composed by instance as column.
        self.Y = Y
        self.n_components = n_components
        ica = FastICA(n_components=n_components)
        self.Y_ = ica.fit_transform(Y.T).T
        self.A_ = ica.mixing_
        self.A_inv = ica.components_
        self.mean_ = ica.mean_[:,np.newaxis]
#        assert np.allclose(Y, np.dot(self.A_,self.Y_)+self.mean_)
        
    def transform(self, bounds, n_instance):
        self.bounds = bounds
        self.n_instance = n_instance
        
        self.center = np.sum(bounds, axis=1)/2
        self.center_ = np.dot(self.A_inv,
                              self.center[:,np.newaxis]-self.mean_*n_instance)        
        self.vertice = ica_constrain.__binary_generator(bounds)
        self.vertice_ = np.dot(self.A_inv, self.vertice-self.mean_*n_instance)
        
        self.bounds_ = np.zeros([self.n_components, 2])
        for n_iter in range(self.n_components):
            candidates = self.vertice_[n_iter]
            left = candidates[candidates <= self.center_[n_iter]]
            right = candidates[candidates >= self.center_[n_iter]]
            self.bounds_[n_iter,0] = left.max()
            self.bounds_[n_iter,1] = right.min()
           
        # only for checking convenience
        self.result['bounds'] = bounds 
        self.result['bounds_'] = self.bounds_
        return self.bounds_
    
    def check(self, Z, name = None):
        if name == None:
            self.check_counter += 1
            name = 'No. ' + str(self.check_counter) +' check instrance'
        # calculation
        Z_sum = np.sum(Z,axis=1)[:,np.newaxis]
        Z_sum_ = np.dot(self.A_inv, Z_sum-self.mean_*self.n_instance)
        diff_left = Z_sum_-self.bounds_[:,0][:,np.newaxis]
        diff_right = self.bounds_[:,1][:,np.newaxis]-Z_sum_
        constrain_indicator = np.logical_and(diff_left>=0, diff_right>=0)
        TorF = diff_left.min() >= 0 and diff_right.max() >= 0
        # recording
        if len(self.result['name']) >= 1:
            self.result['constrain_indicator'] = (
                    np.c_[self.result['constrain_indicator'],constrain_indicator])
            self.result['Z_sum'] = np.c_[self.result['Z_sum'],Z_sum]
            self.result['Z_sum_'] = np.c_[self.result['Z_sum_'],Z_sum_]
            self.result['diff_left'] = np.c_[self.result['diff_left'],diff_left]
            self.result['diff_right'] = np.c_[self.result['diff_right'],diff_right]
        else:
            self.result['constrain_indicator'] = constrain_indicator
            self.result['Z_sum'] = Z_sum
            self.result['Z_sum_'] = Z_sum_
            self.result['diff_left'] = diff_left
            self.result['diff_right'] = diff_right
        self.result['name'].append(name)
        self.result['TorF'].append(TorF)
        return TorF
    
    
    def check_part(self, Z, n_part, name = None):
        # Taking bounds of the first n_part independent compoennts out of all
        # n_components components in the process of the fun 'fit"
        if name == None:
            self.check_counter_self += 1
            name = 'No. ' + str(self.check_counter) +' check_part instrance'
        # calculation(just check first n_part component bounds)
        Z_sum = np.sum(Z,axis=1)[:,np.newaxis]
        Z_sum_ = np.dot(self.A_inv, Z_sum-self.mean_*self.n_instance)
        diff_left = Z_sum_-self.bounds_[:,0][:,np.newaxis]
        diff_right = self.bounds_[:,1][:,np.newaxis]-Z_sum_
        Z_sum_ = Z_sum_[0,n_part]
        diff_left = diff_left[0:n_part]
        diff_right = diff_right[0:n_part]
        constrain_indicator = np.logical_and(diff_left>=0, diff_right>=0)
        TorF = diff_left.min() >= 0 and diff_right.max() >= 0
        # recording
        self.result_part['name'].append(name)
        self.result_part['TorF'].append(TorF)
        if len(self.result_part['name']) >= 1:
            self.result_part['constrain_indicator'] = (
                    np.c_[self.result_part['constrain_indicator'],constrain_indicator])
            self.result_part['Z_sum'] = np.c_[self.result_part['Z_sum'],Z_sum]
            self.result_part['Z_sum_'] = np.c_[self.result_part['Z_sum_'],Z_sum_]
            self.result_part['diff_left'] = np.c_[self.result_part['diff_left'],diff_left]
            self.result_part['diff_right'] = np.c_[self.result_part['diff_right'],diff_right]
        else:
            self.result_part['constrain_indicator'] = constrain_indicator
            self.result_part['Z_sum'] = Z_sum
            self.result_part['Z_sum_'] = Z_sum_
            self.result_part['diff_left'] = diff_left
            self.result_part['diff_right'] = diff_right
        return TorF
        
    def __binary_generator(bounds):
        if bounds.size == 2:
            return np.array([bounds[0,0],bounds[0,1]])[np.newaxis,:]
        else:
            last = ica_constrain.__binary_generator(bounds[1:,:])
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