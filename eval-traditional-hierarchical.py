# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:53:40 2022

@author: katha
"""

import numpy as np
import tensorflow as tf
import random as python_random
import random
import warnings
import pandas as pd
import networkx as nx
import pickle
from scipy import stats
from traditional import synth_BA, synth_ER, synth_WS, synth_2K
import copy
warnings.simplefilter(action='ignore')
def get_edge_attributes(G, name):
    edges = G.edges(data=True)
    return dict( (x[:-1], x[-1][name]) for x in edges if name in x[-1] )

np.random.seed(0)
python_random.seed(0)
tf.random.set_seed(0)

name = "BREN"
clusters = 4
graph_metrics = []
algo ="2K"
load = False
save = True

adj_matrix_real = np.loadtxt("adj_matrices\\"+name+"_weighted.txt")
maximum_dist = np.max(adj_matrix_real)


real = nx.Graph(np.squeeze(adj_matrix_real),data = True)

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

name = name
if clusters == 4:
    paths = [name+"_local_0_c4\\", name+"_local_1_c4\\", name+"_local_2_c4\\", name+"_local_3_c4\\",name+"_global_c4\\"]
if clusters == 3:
    paths = [name+"_local_0_c3\\", name+"_local_1_c3\\", name+"_local_2_c3\\",name+"_global_c3\\"]
if clusters == 2:
    paths = [name+"_local_0_c2\\", name+"_local_1_c2\\", name+"_global_c2\\"]
for k in range(1000):
    full_graph = nx.Graph()
    network__parts = []
    for path in paths:
        if load == True:
            print("Skip synthesizing and load already synthesized topologies.")
            continue
        
        nodes = np.loadtxt(path + "nodes.txt")

        nr_nodes = nodes.size
        
        if "global" in path:
            with open(path+"edges.pkl","rb") as g:
                real_edges = pickle.load(g)

            real_subgraph = nx.Graph(real_edges)
            partial_view = nx.to_numpy_array(real_subgraph)
        else:   
            partial_view = adj_matrix_real[np.ix_(nodes.astype(int), nodes.astype(int))]
        if "local" in path:
            if algo == "2K":
                synth = synth_2K(partial_view)
    
            if algo == "BA":
                synth = synth_BA(partial_view)
    
            if algo == "ER":
                synth = synth_ER(partial_view)    
    
            if algo == "WS":
                synth = synth_WS(partial_view)
              
        elif "global" in path:
            if algo == "2K":
                synth = synth_2K(partial_view, False)
            
                
        mapping = dict(zip(sorted(synth), nodes.astype(int)))
        synth = nx.relabel_nodes(synth, mapping)
        
        full_graph = nx.compose(full_graph,synth)
    
    if "global" in path:
            edges_to_add = []
            added_edges = 0
            components = sorted(nx.connected_components(full_graph), key=len, reverse=True)
            if not nx.is_connected(full_graph):
                for j in range(len(components) - 1):
                    comp1 = components[0]
                    comp2 = components[j+1]
                    node1 = random.sample(comp1, 1)
                    node2 = random.sample(comp2, 1)
                    added_edges = added_edges +1 
                    
                    edges_to_add.append((node1[0], node2[0]   ))
        
            #full_graph.add_edges_from(edges_to_add, weight =  random.choice(list(get_edge_attributes(full_graph, 'weight').values())))
            for edge_to_add in edges_to_add:
                print(edge_to_add)
                full_graph.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(full_graph, 'weight').values())))
            
            r = 0
            while r < added_edges and len(full_graph.edges) >= len(full_graph.nodes):
                edges = list(full_graph.edges)
                testing_graph = copy.deepcopy(full_graph)
                chosen_edge = random.choice(edges)
                testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
                if nx.is_connected(testing_graph):
                    full_graph.remove_edge(chosen_edge[0], chosen_edge[1])
                    r = r + 1
                    print("Succesfully rewired Graph.")

    matrix_norm = nx.to_numpy_array(full_graph)
    full_graph = nx.Graph(matrix_norm)
            
    if (save == True):
        print("Sample saved.")
        with open(path +"/synth_sample_"+algo+"_"+str(k)+".pkl","wb") as f:
            pickle.dump(full_graph,f)
    
    if (load == True):
        print("Sample loaded.")
        with open(path +"/synth_sample_"+algo+"_"+str(k)+".pkl","rb") as f:
            full_graph = pickle.load(f)
    
    links = len(full_graph.edges)
    weights = list(get_edge_attributes(full_graph, 'weight').values())
    
    bcs = list(nx.betweenness_centrality(full_graph).values())
    ccs = list(nx.closeness_centrality(full_graph).values())
    dcs = list(nx.degree_centrality(full_graph).values()) 

    bcsw = list(nx.betweenness_centrality(full_graph,weight='weight').values())
    ccsw = list(nx.closeness_centrality(full_graph, distance = 'weight').values())
    
    ks_weights = (stats.ks_2samp(weights_real, weights))
    ks_bcs = (stats.ks_2samp(bcs_real, bcs))
    ks_ccs = (stats.ks_2samp(ccs_real, ccs))
    ks_dcs = (stats.ks_2samp(dcs_real, dcs))
    ks_bcsw = (stats.ks_2samp(bcsw_real, bcsw))
    ks_ccsw = (stats.ks_2samp(ccsw_real, ccs))
    
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
 
    graph_metrics =  graph_metrics + [[k,links, sum(weights), bcw, ccw, ks_weights[0], ks_weights[1], ks_bcs[0], ks_bcs[1], ks_ccs[0], ks_ccs[1], ks_dcs[0], ks_dcs[1], ks_bcsw[0], ks_bcsw[1], ks_ccsw[0], ks_ccsw[1], bc, cc, dc, ad_weights[0], ad_weights[2], ad_bcs[0], ad_bcs[2], ad_ccs[0], ad_ccs[2], ad_dcs[0], ad_dcs[2], ad_bcsw[0], ad_bcsw[2], ad_ccsw[0], ad_ccsw[2]]]
graph_metrics_df = pd.DataFrame(graph_metrics, columns=["i","Links","Weights","BCW","CCW","Stat_Weights","p-value_Weights","Stat_BC", "p-value_BC","Stat_CC", "p-value_CC","Stat_DC", "p-value_DC","Stat_BCW", "p-value_BCW","Stat_CCW", "p-value_CCW","BC","CC","DC","AD_Weights","p-value_Weights","AD_BC", "p-value_BC","AD_CC", "p-value_CC","AD_DC", "p-value_DC","AD_BCW", "p-value_BCW","AD_CCW", "p-value_CCW"])
graph_metrics_df.to_csv("graph_metrics_"+algo+"_"+name+"_"+str(clusters)+".csv",index=False,sep=";")


print("BC " + str(np.mean(graph_metrics_df["BC"])))
print("CC " + str(np.mean(graph_metrics_df["CC"])))
print("DC " + str(np.mean(graph_metrics_df["DC"])))

print("BCW " + str(np.mean(graph_metrics_df["BCW"])))
print("CCW " + str(np.mean(graph_metrics_df["CCW"])))

        
        
        
        
        