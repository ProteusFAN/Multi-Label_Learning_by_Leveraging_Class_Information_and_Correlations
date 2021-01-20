#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:42:43 2017

@author: Proteus
"""
import numpy as np
from sklearn.decomposition import FastICA

class ica_constrain(object):
    def fit(self, Y, n_components):
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
        return self.bounds_
    
    def check(self, Z):
        self.Z_sum = np.sum(Z,axis=1)[:,np.newaxis]
        self.Z_sum_ = np.dot(self.A_inv, self.Z_sum-self.mean_*self.n_instance)
        self.diff_left = self.Z_sum_-self.bounds_[:,0][:,np.newaxis]
        self.diff_right = self.bounds_[:,1][:,np.newaxis]-self.Z_sum_
        self.constrain_indicator = np.logical_and(self.diff_left>=0,
                                                  self.diff_right>=0)
        return self.diff_left.min() >= 0 and self.diff_right.max() >= 0
    
    def check_part(self, Z, n_part):
        # just check first n_part component bounds
        self.Z_sum = np.sum(Z,axis=1)[:,np.newaxis]
        self.Z_sum_ = np.dot(self.A_inv, self.Z_sum-self.mean_*self.n_instance)
        diff_left = self.Z_sum_-self.bounds_[:,0][:,np.newaxis]
        diff_right = self.bounds_[:,1][:,np.newaxis]-self.Z_sum_
        self.diff_left_part = diff_left[0:n_part]
        self.diff_right_part = diff_right[0:n_part]
        self.constrain_indicator_part = np.logical_and(self.diff_left_part>=0,
                                                  self.diff_right_part>=0)
        return self.diff_left_part.min()>=0 and self.diff_right_part.max()>=0
        
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
        
        
    

        
        
        
        
    