# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:19:01 2022

@author: katha
"""
import os
import tensorflow as tf
import numpy as np
import networkx as nx
from wangan import WANGAN

name = "BREN"
path = name + "_naive"
isExist = os.path.exists(path)
if not isExist:
  os.makedirs(path)

adj_matrix = np.loadtxt("adj_matrices\\"+ name + "_weighted.txt")
G_weighted = nx.from_numpy_matrix(adj_matrix) 

learning_rate = 0.001
epochs = 1000
batch_size = 100 # 50 if GtsSlovakia 100 else
samples = 10000

for seed in range(10):
    WANGAN(path, adj_matrix, batch_size, learning_rate, epochs, samples, False, seed)

