# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:19:01 2022

@author: katha
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx
from sklearn.cluster import SpectralClustering
from wangan import WANGAN
import random
import tensorflow as tf
import pickle

# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(0)
   tf.random.set_seed(0)
   np.random.seed(0)
   random.seed(0)

# Fix seed for reproducibility of clustering, seeds for GAN are reset in WANGAN.py to test different ones
reset_random_seeds()

name = "bren"
clusters = 4 # 2,3,4

for i in range(clusters):
    isExist = os.path.exists(name + "_local_"+ str(i)+"_c"+str(clusters))
    if not isExist:
      os.makedirs(name + "_local_"+ str(i)+"_c"+str(clusters))

isExist = os.path.exists(name + "_global_c"+str(clusters))
if not isExist:
  os.makedirs(name + "_global_c"+str(clusters))

adj_matrix = np.loadtxt("adj_matrices\\"+ name + "_weighted.txt")
G_weighted = nx.from_numpy_matrix(adj_matrix)

sp_matrix = np.asarray(np.zeros(shape=(len(G_weighted),len(G_weighted),)))
for src in G_weighted.nodes:
    for trg in G_weighted.nodes:
        sp_matrix[src][trg] = nx.shortest_path_length(G_weighted, src, trg, weight = 'weight')

maximum_dist = np.max(sp_matrix)
minimum_dist = np.min(sp_matrix) # will always be zero, as the distance from a node to itself is always 0
raw_sp_matrix = sp_matrix
sp_matrix = ((sp_matrix - minimum_dist) / (maximum_dist - minimum_dist))
sim_matrix = copy.deepcopy(sp_matrix)

sim_matrix =  1 - sim_matrix

sc = SpectralClustering(clusters, affinity='precomputed', n_init=100, random_state=0)
sc.fit(sim_matrix)    
labels = sc.labels_

G = nx.from_numpy_matrix(adj_matrix)
nx.draw(G, pos=nx.nx_pydot.graphviz_layout(G), with_labels=True, node_color=labels)  
plt.show()

adj_matrix = np.loadtxt("adj_matrices\\"+ name + "_weighted.txt")
G_weighted = nx.from_numpy_matrix(adj_matrix) 

# Global view first
for seed in range(10):
    # below can actually happen outside the loop and repeating is overhead... ToDo
    nodes = []
    edges = []
    global_view = nx.Graph()
    for i in range(len(G_weighted)):
        for edge in G_weighted.edges(i, data =True):
            print(edge)
            src = edge[0]
            trg = edge[1]
            src_cluster = labels[src]
            trg_cluster = labels[trg]
            if src_cluster != trg_cluster:
                nodes = nodes + [src, trg]
                edges.append(edge)
                             
    global_view.add_edges_from(edges)
    
    # here we maintain the node -> label ordering the way it appeared/was added to the global view graph, so we can map them later again
    # In other words: when transforming to numpy matrix, the first column/row does not necessarily correspond to node with label 1 anymore
    indexes = np.unique(nodes, return_index=True)[1]
    nodes = np.array([nodes[index] for index in sorted(indexes)])
    np.savetxt(name +"_global_c"+str(clusters)+"/nodes.txt", nodes.astype(int), fmt = "%i")

    with open(name +"_global_c"+str(clusters)+"/edges.pkl","wb") as f:
        pickle.dump(edges,f)
       
    network_to_synthesize = nx.to_numpy_matrix(global_view)
     
    learning_rate = 0.0001
    epochs = 200
    batch_size = 50
    samples = 5000
    if len(global_view) >= 16:
        learning_rate = 0.001
  
    WANGAN(name + "_global_c"+str(clusters), network_to_synthesize, batch_size, learning_rate, epochs, samples, True, seed)
    

# Local views, for each 10 runs with differet seeds
for i in range(len(np.unique(labels))):
    for seed in range(10):       
            idx = list(np.where(labels == i))[0] # here nodes are in ascending order by default in the np.matrix, so no mapping like for global needed
            local = adj_matrix[np.ix_(idx, idx)]
            network_to_synthesize = local
            learning_rate = 0.0001
            epochs = 500
            batch_size = 50
            samples = 5000
            if len(local) >= 16:
                learning_rate = 0.001

            np.savetxt(name + "_local_"+ str(i)+"_c"+str(clusters)+"/nodes.txt", idx.astype(int), fmt = "%i")
        
            WANGAN(name + "_local_"+ str(i) +"_c"+str(clusters), network_to_synthesize, batch_size, learning_rate, epochs, samples, True, seed)
        
    
 