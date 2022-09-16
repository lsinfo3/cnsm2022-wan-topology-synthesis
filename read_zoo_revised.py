# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:20:40 2022

@author: katha
"""

import os
import numpy as np
import re
archive = "networks/"

for file in os.listdir(archive):
    if file.endswith("KN.ned"): # could also use HF.ned, does not matter as they are identical.
        network_path = archive + file
        fin = open(network_path, "rt")
        data = fin.read()
        fin.close()

        connections = re.split(r"connections allowunconnected:", data)[1] # just adhering to the normal .ned file structure in the OOS and cut off some unnecessary data
        nodes = len(set(re.findall(r"ofs_[0-9]+", data))) # node nomenclature in the .ned files
        adj_matrix = np.zeros(shape=(nodes,nodes))
        adj_unweighted = np.zeros(shape=(nodes,nodes))
        connections = re.split(r"// placement 1, k = 2", connections)[0] # just adhering to the normal .ned file structure in the OOS and cut off some unnecessary data
        connections_cleaned = re.findall(r"ofs_[0-9]+\.gateDPlane.*distance.* ofs_[0-9]+\.gateDPlane.*",connections)  # edge nomenclature in the .ned files
        for connection in connections_cleaned:
            split = re.split(r"<-->", connection)
            node1 = int(re.findall(r"[0-9]+",split[0])[0]) -1 # start with indexing at 0, .ned files start with 1
            node2 = int(re.findall(r"[0-9]+",split[2])[0]) -1

	        # existing link with 0 geographical distance nonsensical (-> 0 means either identical node or no connection...), just set it to 1 (minimum distance, as all distances in this subset are integers). 
	        # otherwise could maybe potentially lead to problems with calculations with graph metrics i.e. divisions by zero (?) or at least just really really high values for e.g. closeness, also just in general makes life easier for us (e.g. distinguishing between existing links in weighted adj. matrix when processing.)
            # also NetworkX has problems for zero-weights for betweenness according to documentation
            distance = int(re.findall(r"[0-9]+",split[1])[0])
            # distance = np.max(1,distance)
            if distance == 0: 
                print("File "+file+": Link with distance 0 detected, setting to 1 (minimum distance)")
                distance = 1

            adj_matrix[node1][node2] = distance
            adj_matrix[node2][node1] = distance
            
            adj_unweighted[node1][node2] = 1
            adj_unweighted[node2][node1] = 1
            
        np.savetxt('adj_matrices\\'+ re.sub(r"_KN.ned","_weighted",file) + '.txt', adj_matrix)

        np.savetxt('adj_matrices\\'+ re.sub(r"_KN.ned","_unweighted",file) + '.txt', adj_unweighted)


        
