# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:05:13 2022

@author: katha
"""

import numpy as np
import tensorflow as tf
import random as python_random
import copy
import random
import warnings
import networkx as nx
import scipy

warnings.simplefilter(action='ignore', category=FutureWarning)

# https://stackoverflow.com/questions/28107939/assigning-networkx-edge-attributes-weights-to-a-dict
def get_edge_attributes(G, name):
    edges = G.edges(data=True)
    return dict( (x[:-1], x[-1][name]) for x in edges if name in x[-1] )

# fix seeds for reproduceability (not really necessary here as we just define functions here that call from other scripts)
np.random.seed(0)
python_random.seed(0)
tf.random.set_seed(0)


def matrix_to_nx(adj_matrix):

    maximum_dist = np.max(adj_matrix)
    adj_matrix = (adj_matrix / maximum_dist)
    
    perm_unweighted = copy.deepcopy(adj_matrix)
    perm_unweighted[perm_unweighted > 0]=1
    
    real = nx.Graph(np.squeeze(adj_matrix),data = True)
    
    return real

def synth_ER(adj_matrix):
    weights = adj_matrix[adj_matrix != 0]
    real = matrix_to_nx(adj_matrix)
    n = len(real.nodes)
    fully_meshed = scipy.special.binom(n, 2)
    p = len(real.edges)/fully_meshed
    G=nx.erdos_renyi_graph(n,p)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = np.random.choice(weights)
        
    ######## REWIRE ########
    edges_to_add = []
    added_edges = 0
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not nx.is_connected(G):
        for j in range(len(components) - 1):
            comp1 = components[0]
            comp2 = components[j+1]
            node1 = random.sample(comp1, 1)
            node2 = random.sample(comp2, 1)
            added_edges = added_edges + 1       
            edges_to_add.append((node1[0], node2[0]))

    for edge_to_add in edges_to_add:
        G.add_edges_from([edge_to_add],weight= random.choice(list(get_edge_attributes(G, 'weight').values())))
        
    r = 0
    while r < added_edges and len(G.edges) >= len(G.nodes):
        edges = list(G.edges)
        testing_graph = copy.deepcopy(G)
        chosen_edge = random.choice(edges)
        testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
        if nx.is_connected(testing_graph):
            G.remove_edge(chosen_edge[0], chosen_edge[1])
            r = r + 1
    ########################
    
    return G
     
      
def synth_BA(adj_matrix):
    weights = adj_matrix[adj_matrix != 0]
    real = matrix_to_nx(adj_matrix)
    n =  len(real.nodes)
    m=int(round(len(real.edges)/n))
    np.max([m,1])
    G = nx.barabasi_albert_graph(n,m)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = np.random.choice(weights)
        
    ######## REWIRE ########
    edges_to_add = []
    added_edges = 0
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not nx.is_connected(G):
        for j in range(len(components) - 1):
            comp1 = components[0]
            comp2 = components[j+1]
            node1 = random.sample(comp1, 1)
            node2 = random.sample(comp2, 1)
            added_edges = added_edges + 1         
            edges_to_add.append((node1[0], node2[0]))

    for edge_to_add in edges_to_add:
        G.add_edges_from([edge_to_add],weight= random.choice(list(get_edge_attributes(G, 'weight').values())))
          
    r = 0
    while r < added_edges and len(G.edges) >= len(G.nodes):
        edges = list(G.edges)
        testing_graph = copy.deepcopy(G)
        chosen_edge = random.choice(edges)
        testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
        if nx.is_connected(testing_graph):
            G.remove_edge(chosen_edge[0], chosen_edge[1])
            r = r + 1
    ########################

    return G


def synth_WS(adj_matrix):
    weights = adj_matrix[adj_matrix != 0]
    real = matrix_to_nx(adj_matrix)
    n = len(real.nodes)
    k = int(round(len(real.edges)/(n/2)))
    k = np.max([k, 2]) # maximum of two, 1 will be 0, and 0 makes no sense
    print(len(real.edges))
    if k%2 != 0:
        exp_edges = n*(k-1)/2
        exp_edges_alternative = n*(k+1)/2
        if np.abs(len(real.edges) - exp_edges) > np.abs(len(real.edges) - exp_edges_alternative):
            k = k+1
    
    G = nx.watts_strogatz_graph(n,k,0.2)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = np.random.choice(weights)
        
    ######## REWIRE ########
    edges_to_add = []
    added_edges = 0
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not nx.is_connected(G):
        for j in range(len(components) - 1):
            comp1 = components[0]
            comp2 = components[j+1]
            node1 = random.sample(comp1, 1)
            node2 = random.sample(comp2, 1)
            added_edges = added_edges + 1      
            edges_to_add.append((node1[0], node2[0]))

    for edge_to_add in edges_to_add:
        G.add_edges_from([edge_to_add],weight= random.choice(list(get_edge_attributes(G, 'weight').values())))
        
    r = 0
    while r < added_edges and len(G.edges) >= len(G.nodes):
        edges = list(G.edges)
        testing_graph = copy.deepcopy(G)
        chosen_edge = random.choice(edges)
        testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
        if nx.is_connected(testing_graph):
            G.remove_edge(chosen_edge[0], chosen_edge[1])
            r = r + 1
    ########################
 
    return G

def synth_2K(adj_matrix, rewire = True):
    weights = adj_matrix[adj_matrix != 0]
    real = matrix_to_nx(adj_matrix)
    n = len(real.nodes)
    jdd = np.zeros(shape=(n,n))
    jdd_dict = {}
    for edge in list(real.edges):
        a = edge[0]
        b = edge[1]
        a_deg = real.degree[a]
        b_deg = real.degree[b]
        jdd[a_deg][b_deg] = jdd[a_deg][b_deg] + 1

        if a_deg in jdd_dict:
            if b_deg in jdd_dict[a_deg]:
                jdd_dict[a_deg][b_deg] = jdd_dict[a_deg][b_deg] + 1
                
        if a_deg in jdd_dict:
            if not b_deg in jdd_dict[a_deg]:
                jdd_dict[a_deg][b_deg] = 1
                
        if not a_deg in jdd_dict:
            jdd_dict[a_deg] = {b_deg: 1 }

    # switch around a and b to count both directions
    for edge in list(real.edges):
        b = edge[0]
        a = edge[1]
        a_deg = real.degree[a]
        b_deg = real.degree[b]
        jdd[a_deg][b_deg] = jdd[a_deg][b_deg] + 1

        if a_deg in jdd_dict:
            if b_deg in jdd_dict[a_deg]:
                jdd_dict[a_deg][b_deg] = jdd_dict[a_deg][b_deg] + 1
                
        if a_deg in jdd_dict:
            if not b_deg in jdd_dict[a_deg]:
                jdd_dict[a_deg][b_deg] = 1
                
        if not a_deg in jdd_dict:
            jdd_dict[a_deg] = {b_deg: 1 }

    G = nx.joint_degree_graph(jdd_dict)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = np.random.choice(weights)
    
    ######## REWIRE ########
    if rewire:
        edges_to_add = []
        added_edges = 0
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        if not nx.is_connected(G):
            for j in range(len(components) - 1):
                comp1 = components[0]
                comp2 = components[j+1]
                node1 = random.sample(comp1, 1)
                node2 = random.sample(comp2, 1)
                added_edges = added_edges + 1 
                edges_to_add.append((node1[0], node2[0]))

        for edge_to_add in edges_to_add:
            G.add_edges_from([edge_to_add],weight= random.choice(list(get_edge_attributes(G, 'weight').values())))
        
        r = 0
        while r < added_edges and len(G.edges) >= len(G.nodes):
            edges = list(G.edges)
            testing_graph = copy.deepcopy(G)
            chosen_edge = random.choice(edges)
            testing_graph.remove_edge(chosen_edge[0], chosen_edge[1])
            if nx.is_connected(testing_graph):
                G.remove_edge(chosen_edge[0], chosen_edge[1])
                r = r + 1
    ########################
    
    return G
