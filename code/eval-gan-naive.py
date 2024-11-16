# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:43:20 2022

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
from itertools import compress

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# https://stackoverflow.com/questions/28107939/assigning-networkx-edge-attributes-weights-to-a-dict
def get_edge_attributes(G, name):
    edges = G.edges(data=True)
    return dict( (x[:-1], x[-1][name]) for x in edges if name in x[-1] )


names = ["GTSSLOVAKIA", "BREN", "GEANT2001", "BTNORTHAMERICA"] # one of GEANT2001, BREN, BTNORTHAMERICA, GTSSLOVAKIA
load = False
save = True
sample_weights_list = [False, False, True]
colors = ["RGB", "BW", "BW"]

for name in names:
    for sample_weights, color in zip(sample_weights_list, colors):
        
        # fix seeds for reproduceability, so that loop iterations are not dependent in terms of random numbers...
        np.random.seed(0)
        python_random.seed(0)
        tf.random.set_seed(0)
        
        graph_metrics = []
        
        # load real matrix from adj_matrices folder
        adj_matrix_real = np.loadtxt("adj_matrices\\"+name+"_weighted.txt")
        maximum_dist = np.max(adj_matrix_real)
        
        ####### plotting the weighted and unweighted adjancency matrices #######
        perm_unweighted = copy.deepcopy(adj_matrix_real)
        perm_unweighted[perm_unweighted > 0]=1
        
        network_size = len(perm_unweighted)
        
        # RGB
        rgb = np.stack((np.asarray((perm_unweighted)),np.asarray((adj_matrix_real/maximum_dist))), axis=-1)
        rgb = np.dstack((np.asarray(rgb),np.asarray(np.zeros(shape=(network_size,network_size,1)))))
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.imshow(rgb)
        plt.savefig(name+'_rgb.pdf',bbox_inches='tight',pad_inches=0.1)
        plt.show()
        
        # BW
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.imshow(perm_unweighted, cmap='gray_r')
        plt.savefig(name+'_bw.pdf',bbox_inches='tight',pad_inches=0.1)
        plt.show()
        ############################
        
        
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
        
        path = name+"_naive\\"
        
        for k in range(100):
            for s in range(10):
                nr_nodes = len(adj_matrix_real)
                if load == True:
                    print("Skip synthesizing and load already synthesized topologies.")       
                
                if load == False:
                    with open(path+"sample_at_epoch_1000_"+str(k)+"_"+str(s)+"_"+color+".pkl","rb") as f:
                        adj_matrix = pickle.load(f)
                        adj_matrix = adj_matrix[0:nr_nodes,0:nr_nodes] # this basically cuts off the padding
             
                        extracted_unweighted = np.asarray(np.zeros(shape=(len(adj_matrix),len(adj_matrix),1))) # extract the "layer"/colorchannel that depicts if their exists a link or not (exists for for BW and RGB)
                        for i in range(len(adj_matrix)):
                            for j in range(len(adj_matrix)):
        		            	# choose if link exists according to Bernoulli distribution
                                sum_entries = adj_matrix[i][j][0] + adj_matrix[j][i][0]
                                decision = np.random.choice([0,1], p=[1-(sum_entries/2),(sum_entries/2)])
                                extracted_unweighted[i][j][0] = decision
                                extracted_unweighted[j][i][0] = decision
                                if (i == j):
                                    extracted_unweighted[i][j][0] = 0.0 # no self-loops
                        
                        if adj_matrix.shape == (nr_nodes, nr_nodes, 3): # if we have an RGB image, i.e., 3 colorchannels
                            
                            extracted_weighted = np.asarray(np.zeros(shape=(len(adj_matrix),len(adj_matrix),1)))
                            for i in range(len(extracted_unweighted)):
                                for j in range(len(extracted_unweighted)):
                                    avg_dist =  (adj_matrix[i][j][1] +  adj_matrix[j][i][1])/2 # link weight as average of the symmetric entries, if the decided above that the link exists                           
                                    if extracted_unweighted[i][j][0] ==1.0 and extracted_unweighted[j][i][0] ==1.0:
                                        extracted_weighted[i][j][0] = np.max([avg_dist, 1/maximum_dist]) # 1 is the minimum distance, so we can cap it off there/smooth it out
                                        extracted_weighted[j][i][0] = np.max([avg_dist, 1/maximum_dist])
                                        
                            extracted_weighted = extracted_weighted * maximum_dist # rescale the networks, as it was normalized during synthesis
                            extracted_weighted = np.squeeze(extracted_weighted)
                            full_graph = nx.Graph(extracted_weighted,data = True)
                            
                            ######## REWIRE ########
                            edges_to_add = [] # rewire the graph so that is connected
                            added_edges = 0
                            components = sorted(nx.connected_components(full_graph), key=len, reverse=True) # connect all smaller components/isolated nodes to the biggest component
                            if not nx.is_connected(full_graph):
                                for j in range(len(components) - 1):
                                    comp1 = components[0]
                                    comp2 = components[j+1]
                                    node1 = random.sample(comp1, 1)
                                    node2 = random.sample(comp2, 1)
                                    added_edges = added_edges +1                            
                                    edges_to_add.append((node1[0], node2[0]   ))
                            
                            for edge_to_add in edges_to_add:
        		            	# edge weight is chosen from already inferred weights, so we do not utilize any other information, only the ones that the GAN provides
                                full_graph.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(full_graph, 'weight').values())))
                            
                            # now we basically cap some links, so we rewired the graph in the end
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
        
                        else: # if we have BW image
                            if sample_weights: # if we want to sample on weights onto BW image
                                weights = adj_matrix_real[adj_matrix_real != 0] # get list of real weights from real matrix
                                extracted_weighted = np.asarray(np.zeros(shape=(len(adj_matrix),len(adj_matrix),1)))
                                for i in range(len(extracted_unweighted)):
                                    for j in range(len(extracted_unweighted)):
                                        if extracted_unweighted[i][j][0] ==1.0 and extracted_unweighted[j][i][0] ==1.0:
                                                extracted_weighted[i][j][0] = np.random.choice(weights) # actually sample the weights if link exists
                                                extracted_weighted[j][i][0] = np.random.choice(weights)
                                        
                                extracted_weighted = np.squeeze(extracted_weighted)
                                full_graph = nx.Graph(extracted_weighted,data = True)
                            else:
                                extracted_unweighted = np.squeeze(extracted_unweighted) # no sampling here
                                full_graph = nx.Graph(extracted_unweighted,data = True)
                            
        		            # now rewire again (same as for RGB)
                            ######## REWIRE ########
                            edges_to_add = []
                            added_edges = 0
                            components = sorted(nx.connected_components(full_graph), key=len, reverse=True)
                            if not nx.is_connected(full_graph):
                                for j in range(len(components) - 1):
                                    comp1 = components[0]
                                    comp2 = components[j+1]
                                    node1 = random.sample(comp1, 1)
                                    node2 = random.sample(comp2, 1)
                                    added_edges = added_edges + 1 
                                    
                                    edges_to_add.append((node1[0], node2[0]))
            
                            if sample_weights:          
                                for edge_to_add in edges_to_add:
                                    full_graph.add_edges_from([edge_to_add], weight = random.choice(list(get_edge_attributes(full_graph, 'weight').values())))
                                
                            else: 
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
                
                graph_metrics =  graph_metrics + [[ k,links, sum(weights), bcw, ccw, ks_weights[0], ks_weights[1], ks_bcs[0], ks_bcs[1], ks_ccs[0], ks_ccs[1], ks_dcs[0], ks_dcs[1], ks_bcsw[0], ks_bcsw[1], ks_ccsw[0], ks_ccsw[1], bc, cc, dc, ad_weights[0], ad_weights[2], ad_bcs[0], ad_bcs[2], ad_ccs[0], ad_ccs[2], ad_dcs[0], ad_dcs[2], ad_bcsw[0], ad_bcsw[2], ad_ccsw[0], ad_ccsw[2]]]
        
        graph_metrics_df = pd.DataFrame(graph_metrics, columns=["i","Links","Weights","BCW","CCW","Stat_Weights","p-value_Weights","Stat_BC", "p-value_BC","Stat_CC", "p-value_CC","Stat_DC", "p-value_DC","Stat_BCW", "p-value_BCW","Stat_CCW", "p-value_CCW","BC","CC","DC","AD_Weights","p-value_Weights","AD_BC", "p-value_BC","AD_CC", "p-value_CC","AD_DC", "p-value_DC","AD_BCW", "p-value_BCW","AD_CCW", "p-value_CCW"])
        
        if sample_weights:
            graph_metrics_df.to_csv("graph_metrics_"+name+"_hybrid.csv",index=False,sep=";")
        elif color == "BW":
             graph_metrics_df.to_csv("graph_metrics_"+name+"_BW.csv",index=False,sep=";")
        elif color == "RGB":
             graph_metrics_df.to_csv("graph_metrics_"+name+"_RGB.csv",index=False,sep=";")
             
        print("BC " + str(np.mean(graph_metrics_df["BC"])))
        print("CC " + str(np.mean(graph_metrics_df["CC"])))
        print("DC " + str(np.mean(graph_metrics_df["DC"])))
         
        print("BCW " + str(np.mean(graph_metrics_df["BCW"])))
        print("CCW " + str(np.mean(graph_metrics_df["CCW"])))
        
