# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:38:07 2022

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
warnings.simplefilter(action='ignore')

# https://stackoverflow.com/questions/28107939/assigning-networkx-edge-attributes-weights-to-a-dict
def get_edge_attributes(G, name):
    edges = G.edges(data=True)
    return dict( (x[:-1], x[-1][name]) for x in edges if name in x[-1] )

names = ["GTSSLOVAKIA", "BREN", "GEANT2001", "BTNORTHAMERICA"]
algos = ["2K", "BA", "ER", "WS"]

load = False
save = True
for name in names:
    for algo in algos:
        
        # set seed for every algo, so that loop iterations a not dependent in terms of random numbers...
        np.random.seed(0)
        python_random.seed(0)
        tf.random.set_seed(0)
        
        graph_metrics = []
        
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
        
        path = name+"_naive/"
        
        for k in range(1000):
            
            if load == True:
                print("Skip synthesizing and load already synthesized topologies.")      
        
            if load == False:
                if algo == "2K":
                    full_graph = synth_2K(adj_matrix_real)
            
                if algo == "BA":
                    full_graph = synth_BA(adj_matrix_real)
            
                if algo == "ER":
                    full_graph = synth_ER(adj_matrix_real)    
            
                if algo == "WS":
                    full_graph = synth_WS(adj_matrix_real)    
        
            if (save == True):
                with open(path +"synth_sample_"+algo+"_"+str(k)+"_.pkl","wb") as f:
                    pickle.dump(full_graph,f)
            
            if (load == True):
                with open(path +"synth_sample_"+algo+"_"+str(k)+"_.pkl","rb") as f:
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
            ###################################
            
            graph_metrics =  graph_metrics + [[k,links, sum(weights), bcw, ccw, ks_weights[0], ks_weights[1], ks_bcs[0], ks_bcs[1], ks_ccs[0], ks_ccs[1], ks_dcs[0], ks_dcs[1], ks_bcsw[0], ks_bcsw[1], ks_ccsw[0], ks_ccsw[1], bc, cc, dc, ad_weights[0], ad_weights[2], ad_bcs[0], ad_bcs[2], ad_ccs[0], ad_ccs[2], ad_dcs[0], ad_dcs[2], ad_bcsw[0], ad_bcsw[2], ad_ccsw[0], ad_ccsw[2]]]
      
        graph_metrics_df = pd.DataFrame(graph_metrics, columns=["i","Links","Weights","BCW","CCW","Stat_Weights","p-value_Weights","Stat_BC", "p-value_BC","Stat_CC", "p-value_CC","Stat_DC", "p-value_DC","Stat_BCW", "p-value_BCW","Stat_CCW", "p-value_CCW","BC","CC","DC","AD_Weights","p-value_Weights","AD_BC", "p-value_BC","AD_CC", "p-value_CC","AD_DC", "p-value_DC","AD_BCW", "p-value_BCW","AD_CCW", "p-value_CCW"])
        graph_metrics_df.to_csv("graph_metrics_"+algo+"_"+name+".csv",index=False,sep=";")
        
        
        print("BC " + str(np.mean(graph_metrics_df["BC"])))
        print("CC " + str(np.mean(graph_metrics_df["CC"])))
        print("DC " + str(np.mean(graph_metrics_df["DC"])))
  
        print("BCW " + str(np.mean(graph_metrics_df["BCW"])))
        print("CCW " + str(np.mean(graph_metrics_df["CCW"])))
        

            
            