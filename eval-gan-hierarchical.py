# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:46:47 2022

@author: katha
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random as python_random
import copy
import seaborn as sns
import random
import warnings
import pandas as pd
import networkx as nx
import pickle
from scipy import stats
import scipy
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')

# https://stackoverflow.com/questions/28107939/assigning-networkx-edge-attributes-weights-to-a-dict
def get_edge_attributes(G, name):
    edges = G.edges(data=True)
    return dict( (x[:-1], x[-1][name]) for x in edges if name in x[-1] )

np.random.seed(0)
python_random.seed(0)
tf.random.set_seed(0)

name = "BREN" # one of GEANT2001, BREN, BTNORTHAMERICA, GTSSLOVAKIA
clusters = 2 # 2,3,4
graph_metrics = []
load = False
save = True
sample_weights = False
color = "RGB"

# load real matrix from adj_matrices folder
adj_matrix_real = np.loadtxt("adj_matrices\\"+name+"_weighted.txt")
maximum_dist = np.max(adj_matrix_real)
real = nx.Graph(np.squeeze(adj_matrix_real),data = True)

####### graph metrics of real network #######
links_real = len(real.edges)
weights_real = list(get_edge_attributes(real, 'weight').values())

bcs_real = list(nx.betweenness_centrality(real).values())
ccs_real = list(nx.closeness_centrality(real).values())
dcs_real = list(nx.degree_centrality(real).values()) 

bcsw_real = list(nx.betweenness_centrality(real,weight='weight').values())
ccsw_real = list(nx.closeness_centrality(real, distance = 'weight').values())

bc_real = np.mean(bcs_real)
cc_real = np.mean(ccs_real)
dc_real = np.mean(dcs_real)

bcw_real = np.mean(bcsw_real)
ccw_real = np.mean(ccsw_real)
##########################################

if clusters == 4:
    paths = [name+"_local_0_c4\\", name+"_local_1_c4\\", name+"_local_2_c4\\", name+"_local_3_c4\\",name+"_global_c4\\"]
if clusters == 3:
    paths = [name+"_local_0_c3\\", name+"_local_1_c3\\", name+"_local_2_c3\\",name+"_global_c3\\"]
if clusters == 2:
    paths = [name+"_local_0_c2\\", name+"_local_1_c2\\", name+"_global_c2\\"]

for k in range(100):    
    for s in range(10):
        full_graph = nx.Graph() # new graph where components are added one by one
        for path in paths:
            path = name + "_hierarchical/" + path
            if load == True:
                continue
            nodes = np.loadtxt(path + "nodes.txt")
    
            nr_nodes = nodes.size
            epochs = 500
            if "global" in path:
                epochs = 200
            with open(path+"sample_at_epoch_"+str(epochs)+"_"+str(k)+"_"+str(s)+"_"+color+".pkl","rb") as f:
                adj_matrix = pickle.load(f)
                adj_matrix = adj_matrix[0:nr_nodes,0:nr_nodes] # cut off padding

                # empty matrix to fill
                extracted_unweighted = np.asarray(np.zeros(shape=(len(adj_matrix),len(adj_matrix),1)))
                
                # same as for the naive/flat approach: decide via Bernoulli distribution which links are actually taken
                for i in range(len(adj_matrix)):
                    for j in range(len(adj_matrix)):
                        sum_entries = adj_matrix[i][j][0] + adj_matrix[j][i][0]
                        decision = np.random.choice([0,1], p=[1-(sum_entries/2),(sum_entries/2)])
                        extracted_unweighted[i][j][0] = decision
                        extracted_unweighted[j][i][0] = decision
                        if (i == j):
                            extracted_unweighted[i][j][0] = 0.0
    
                # it would probably be easier if the GAN returned the unnormalized samples directly, instead of the min-max normalized ones 
		        # then we would not need to have the following if-else statement to rescale, if we directly scaled the samples when saving them in the GAN
                if "global" in path: # if global in path we cannot just use the partial adjacency matrix, as there might be too many links
                    partial_weights = []
                    with open(path+"edges.pkl","rb") as g:
                        real_edges = pickle.load(g)
    
                    real_max_dist=0
                    for m in range(len(real_edges)):                   
                        weight = real_edges[m][2]['weight']
                        partial_weights.append(weight)
                        if weight > real_max_dist:
                            real_max_dist = weight
                else: # but if we have local in path, we can just load the partial matrix via their indexes from the saved nodes.txt
                    partial_matrix = adj_matrix_real[np.ix_(nodes.astype(int), nodes.astype(int))]
                    partial_weights = partial_matrix[partial_matrix != 0]
                    real_max_dist = np.max(partial_matrix)
                    
                    
                if adj_matrix.shape == (nr_nodes, nr_nodes, 3): # if RGB
                    
                    extracted_weighted = np.asarray(np.zeros(shape=(len(adj_matrix),len(adj_matrix),1)))
                    
                    for i in range(len(extracted_unweighted)):
                        for j in range(len(extracted_unweighted)):
                            # take avg dist. if linke exists
                            avg_dist =  (adj_matrix[i][j][1] +  adj_matrix[j][i][1])/2
                            if extracted_unweighted[i][j][0] == 1.0 and extracted_unweighted[j][i][0] ==1.0:
                                extracted_weighted[i][j][0] = np.max([avg_dist, 1/real_max_dist])
                                extracted_weighted[j][i][0] = np.max([avg_dist, 1/real_max_dist])
                                
                    extracted_weighted = extracted_weighted * real_max_dist # rescale            
                    extracted_weighted = np.squeeze(extracted_weighted)
                    synth = nx.Graph(extracted_weighted,data = True)
        
                    mapping = dict(zip(sorted(synth), nodes.astype(int))) # map indices of partial matrix to actual nodel labels in full graph
                    synth = nx.relabel_nodes(synth, mapping)

                    # if local in path we basically do it the same like in the flat approach, load partial matrix and rewire if not connected
                    # as we know the local views have to be connected
                    if "local" in path:
                        
                        ######## REWIRE ########
                        edges_to_add = []
                        added_edges = 0
                        components = sorted(nx.connected_components(synth), key=len, reverse=True)
                        if not nx.is_connected(synth):
                            for j in range(len(list(nx.connected_components(synth))) - 1):
                                    comp1 = components[0]
                                    comp2 = components[j+1]
                                    node1 = random.sample(comp1, 1)
                                    node2 = random.sample(comp2, 1)
                                    added_edges = added_edges +1 
                                    
                                    edges_to_add.append((node1[0], node2[0]   ))
            
            
                        if len(extracted_weighted[extracted_weighted>0]) == 0: # shouldn't be the case 
                            synth.add_edges_from(edges_to_add, weight = 1)
                        else:
                            for edge_to_add in edges_to_add:
                                synth.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(synth, 'weight').values())))
                            

                        r = 0
                        while r < added_edges and len(synth.edges) >= len(synth.nodes):
                            edges = list(synth.edges)
                            testing_graph = copy.deepcopy(synth)
                            chosen_edge = random.choice(edges)
                            testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
                            if nx.is_connected(testing_graph):
                                synth.remove_edge(chosen_edge[0], chosen_edge[1])
                                r = r + 1                                
                        ########################
                    
                else: # if BW

                    if sample_weights:
                        weights = partial_weights
                        extracted_weighted = np.asarray(np.zeros(shape=(len(adj_matrix),len(adj_matrix),1)))

                        for i in range(len(extracted_unweighted)):
                            for j in range(len(extracted_unweighted)):
                                if extracted_unweighted[i][j][0] ==1.0 and extracted_unweighted[j][i][0] ==1.0:
                                        extracted_weighted[i][j][0] = np.random.choice(weights)
                                        extracted_weighted[j][i][0] = np.random.choice(weights)
                                
                        extracted_weighted = np.squeeze(extracted_weighted)
                        synth = nx.Graph(extracted_weighted,data = True)
                    else:
                        extracted_unweighted = np.squeeze(extracted_unweighted)
                        synth = nx.Graph(extracted_unweighted,data = True)

                        
                    mapping = dict(zip(sorted(synth), nodes.astype(int)))
                    synth = nx.relabel_nodes(synth, mapping)
                    
                    # if local in path we basically do it the same like in the flat approach, load partial matrix and rewire if not connected
                    # as we know the local views have to be connected
                    if "local" in path:
                        
                        ######## REWIRE ########
                        edges_to_add = []
                        added_edges = 0
                        components = sorted(nx.connected_components(synth), key=len, reverse=True)
                        if not nx.is_connected(synth):
                            for j in range(len(list(nx.connected_components(synth))) - 1):
                                    comp1 = components[0]
                                    comp2 = components[j+1]
                                    node1 = random.sample(comp1, 1)
                                    node2 = random.sample(comp2, 1)
                                    added_edges = added_edges + 1 
                                    edges_to_add.append((node1[0], node2[0]))
            
            
                        if sample_weights:
                            if len(extracted_weighted[extracted_weighted>0]) == 0: # shouldn't be the case 
                                synth.add_edges_from(edges_to_add, weight = 1)
                            else:
                                for edge_to_add in edges_to_add:
                                    synth.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(synth, 'weight').values())))
                            
                        else: 
                            synth.add_edges_from(edges_to_add, weight = 1)
                            
                        r = 0
                        while r < added_edges and len(synth.edges) >= len(synth.nodes):
                            edges = list(synth.edges)
                            testing_graph = copy.deepcopy(synth)
                            chosen_edge = random.choice(edges)
                            testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
                            if nx.is_connected(testing_graph):
                                synth.remove_edge(chosen_edge[0], chosen_edge[1])
                                r = r + 1
                        ########################
               
                
                full_graph = nx.compose(full_graph,synth) # add component to graph
               
        # if global (last element in path-list), all components have been added
        # remember: only the local view have been rewired to be connected
        # as the global view is not necessarily connected -> rewire only after global view is added and graph still disconnected
        if "global" in path:
            
            ######## REWIRE ########
            edges_to_add = []
            added_edges = 0
            components = sorted(nx.connected_components(full_graph), key=len, reverse=True)

            if not nx.is_connected(full_graph):
                for j in range(len(list(nx.connected_components(full_graph))) - 1):
                        comp1 = components[0]
                        comp2 = components[j+1]
                        node1 = random.sample(comp1, 1)
                        node2 = random.sample(comp2, 1)
                        added_edges = added_edges + 1 
                        edges_to_add.append((node1[0], node2[0]))
    
    
            if len(extracted_weighted[extracted_weighted>0]) == 0: # shouldn't happen
                full_graph.add_edges_from(edges_to_add, weight = 1)
            elif color == "BW" and sample_weights:
                for edge_to_add in edges_to_add:
                    full_graph.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(full_graph, 'weight').values())))
                
            elif color == "RGB":
                for edge_to_add in edges_to_add:
                    full_graph.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(full_graph, 'weight').values())))
            
            elif color == "BW" and not sample_weights:  
                full_graph.add_edges_from(edges_to_add, weight = 1)
                
            
            r = 0
            while r < added_edges and len(full_graph.edges) >= len(full_graph.nodes):
                edges = list(full_graph.edges)
                testing_graph = copy.deepcopy(full_graph)
                chosen_edge = random.choice(edges)
                testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
                if nx.is_connected(testing_graph):
                    full_graph.remove_edge(chosen_edge[0], chosen_edge[1])
                    r = r + 1
            ########################
                    
    
        matrix_norm = nx.to_numpy_array(full_graph)
        full_graph = nx.Graph(matrix_norm)
        
        if (save == True):
            with open(path +"/synth_sample_"+str(k)+"_"+str(s)+"_"+color+"_"+str(int(sample_weights == True))+"_.pkl","wb") as f:
                pickle.dump(full_graph,f)
        
        if (load == True):
            with open(path +"/synth_sample_"+str(k)+"_"+str(s)+"_"+color+"_"+str(int(sample_weights == True))+"_.pkl","rb") as f:
                full_graph = pickle.load(f)
        
        ####### graph metrics of synthetic network #######
        links = len(full_graph.edges)
        weights = list(get_edge_attributes(full_graph, 'weight').values())
        
        bcs = list(nx.betweenness_centrality(full_graph).values())
        ccs = list(nx.closeness_centrality(full_graph).values())
        dcs = list(nx.degree_centrality(full_graph).values()) 
    
        bcsw = list(nx.betweenness_centrality(full_graph,weight='weight').values())
        ccsw = list(nx.closeness_centrality(full_graph, distance = 'weight').values())
        
        # Kolgomorov-Smirnov test real vs synth
        ks_weights = (stats.ks_2samp(weights_real, weights))
        ks_bcs = (stats.ks_2samp(bcs_real, bcs))
        ks_ccs = (stats.ks_2samp(ccs_real, ccs))
        ks_dcs = (stats.ks_2samp(dcs_real, dcs))
        ks_bcsw = (stats.ks_2samp(bcsw_real, bcsw))
        ks_ccsw = (stats.ks_2samp(ccsw_real, ccs))
        
        # Anderson-Darling test real vs synth
        ad_weights = (stats.anderson_ksamp([weights_real, weights]))
        ad_bcs = (stats.anderson_ksamp([bcs_real, bcs]))
        ad_ccs = (stats.anderson_ksamp([ccs_real, ccs]))
        ad_dcs = (stats.anderson_ksamp([dcs_real, dcs]))
        ad_bcsw = (stats.anderson_ksamp([bcsw_real, bcsw]))
        ad_ccsw = (stats.anderson_ksamp([ccsw_real, ccs]))
       
        bc = np.mean(bcs)
        cc = np.mean(ccs)
        dc = np.mean(dcs)
        
        bcw = np.mean(bcsw)
        ccw = np.mean(ccsw)
        ##########################################
        
        graph_metrics =  graph_metrics + [[k,links, sum(weights), bcw, ccw, ks_weights[0], ks_weights[1], ks_bcs[0], ks_bcs[1], ks_ccs[0], ks_ccs[1], ks_dcs[0], ks_dcs[1], ks_bcsw[0], ks_bcsw[1], ks_ccsw[0], ks_ccsw[1], bc, cc, dc, ad_weights[0], ad_weights[2], ad_bcs[0], ad_bcs[2], ad_ccs[0], ad_ccs[2], ad_dcs[0], ad_dcs[2], ad_bcsw[0], ad_bcsw[2], ad_ccsw[0], ad_ccsw[2]]]
       
graph_metrics_df = pd.DataFrame(graph_metrics, columns=["i","Links","Weights","BCW","CCW","Stat_Weights","p-value_Weights","Stat_BC", "p-value_BC","Stat_CC", "p-value_CC","Stat_DC", "p-value_DC","Stat_BCW", "p-value_BCW","Stat_CCW", "p-value_CCW","BC","CC","DC","AD_Weights","p-value_Weights","AD_BC", "p-value_BC","AD_CC", "p-value_CC","AD_DC", "p-value_DC","AD_BCW", "p-value_BCW","AD_CCW", "p-value_CCW"])

# distinguish different modes when saving
if sample_weights:
    graph_metrics_df.to_csv("graph_metrics_"+name+"_hybrid"+"_"+str(clusters)+".csv",index=False,sep=";")
elif color == "BW":
     graph_metrics_df.to_csv("graph_metrics_"+name+"_BW"+"_"+str(clusters)+".csv",index=False,sep=";")
elif color == "RGB":
     graph_metrics_df.to_csv("graph_metrics_"+name+"_RGB"+"_"+str(clusters)+".csv",index=False,sep=";")

print("BC " + str(np.mean(graph_metrics_df["BC"])))
print("CC " + str(np.mean(graph_metrics_df["CC"])))
print("DC " + str(np.mean(graph_metrics_df["DC"])))
   
print("BCW " + str(np.mean(graph_metrics_df["BCW"])))
print("CCW " + str(np.mean(graph_metrics_df["CCW"])))