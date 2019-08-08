# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:45:46 2019

@author: 15250
"""
import tensorflow as tf
import numpy as np

#creat date
def generate(size):
    x_data=np.random.rand(100).astype(np.float32)
    #print(x_data)
    y_data=[[3*x**3+0.3] for x in x_data]
    x_data=[[100*x] for x in x_data]
    return x_data,y_data