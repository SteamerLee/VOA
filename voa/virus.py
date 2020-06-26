#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         virus
 Description:  
 Author:       Samuel Li, Ray Li
 Date:         21/06/2020
---------------------------------
'''

import numpy as np


class Virus:

    def __init__(self, dim, bound, structure, type='common'):
        self.dim = dim
        self.bound = bound
        self.fitness = None
        self.type = type
        self.structure = structure
        self.sigma = np.random.rand(self.dim)

    def update_sigma(self, t, t_):
        xx = t * np.random.randn(self.dim) + t_ * np.random.randn(self.dim)
        self.sigma = self.sigma + np.exp(xx)
