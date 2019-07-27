import sklearn
from sklearn.externals import joblib
from bionev.utils import load_embedding
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import networkx as nx
import time

def find_new_relations(*, entity, saved_model, node_mapping_file, embeddings_file, graph_edgelist=None, entity_type=None, k=30):
    embeddings = load_embedding(embeddings_file)
    node_mapping = pd.read_csv(node_mapping_file, sep=',')
    graph = nx.read_edgelist(graph_edgelist)
    model = joblib.load(saved_model)
    node_id = node_mapping.loc[node_mapping["NodeName"] == entity, "NodeID"].iloc[0]
    node1 = embeddings[str(node_id)]
    if entity_type == 'chemical':
        node_list = find_chemicals(entity=entity, embeddings=embeddings, node_mapping=node_mapping, graph=graph)
    elif entity_type == 'phenotype':
        node_list = find_phenotypes(entity=entity, embeddings=embeddings, node_mapping=node_mapping, graph=graph)
    elif entity_type == 'target':
        node_list = find_targets(entity=entity, embeddings=embeddings, node_mapping=node_mapping, graph=graph)
    else:
        X=[]
        node_list = []
        for node, vector in embeddings.items():
            if node == entity:
                continue
            if graph != None:
                if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                    continue
            node2 = np.array(embeddings[node])
            x = node1 * node2
            X.append(x.tolist())
            node_list.append(node)
    prediction_list = get_probabilities(node_list=node_list, model=model, k=k)
    return prediction_list

def find_chemicals(*, entity, embeddings, graph=None, node_mapping):
    X=[]
    node_list = []
    for node, vector in embeddings.items():
        if node == entity:
            continue
        if graph != None:
            if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                continue
        namespace = node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeNamespace"].iloc[0]
        if namespace != 'pubchem':
            continue
        node2 = np.array(embeddings[node])
        x = node1 * node2
        X.append(x.tolist())
        node_list.append(node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeName"].iloc[0])
    return node_list

def find_targets(*, entity, embeddings, graph=None, node_mapping):
    X=[]
    node_list = []
    for node, vector in embeddings.items():
        if node == entity:
            continue
        if graph != None:
            if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                continue
        namespace = node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeNamespace"].iloc[0]
        if namespace != 'uniprot':
            continue
        node2 = np.array(embeddings[node])
        x = node1 * node2
        X.append(x.tolist())
        node_list.append(node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeName"].iloc[0])
    return node_list

def find_phenotypes(*, entity, embeddings, graph=None, node_mapping):
    X=[]
    node_list = []
    for node, vector in embeddings.items():
        if node == entity:
            continue
        if graph != None:
            if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                continue
        namespace = node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeNamespace"].iloc[0]
        if namespace != 'umls':
            continue
        node2 = np.array(embeddings[node])
        x = node1 * node2
        X.append(x.tolist())
        node_list.append(node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeName"].iloc[0])
    return node_list

def get_probabilities(*, node_list, model, k):
    all_prob = {}
    prob_list = model.predict_proba(X)[:,1]
    for i in range(len(node_list)):
        all_prob[node_list[i]] = prob_list[i]
    sorted_list = sorted(all_prob.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_list[:k]
