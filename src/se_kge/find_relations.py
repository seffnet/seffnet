import numpy as np
from tqdm import tqdm

def find_new_relations(*, entity, saved_model, node_mapping, embeddings, graph=None, entity_type=None, k=30):
    if entity_type == 'chemical':
        node_list, relations_list = find_chemicals(entity=entity, embeddings=embeddings, node_mapping=node_mapping, graph=graph)
    elif entity_type == 'phenotype':
        node_list, relations_list = find_phenotypes(entity=entity, embeddings=embeddings, node_mapping=node_mapping, graph=graph)
    elif entity_type == 'target':
        node_list, relations_list = find_targets(entity=entity, embeddings=embeddings, node_mapping=node_mapping, graph=graph)
    else:
        relations_list=[]
        node_list = []
        node_id = node_mapping.loc[node_mapping["NodeName"] == entity, "NodeID"].iloc[0]
        node1 = embeddings[str(node_id)]
        for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
            if node == entity:
                continue
            if graph != None:
                if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                    continue
            relation = node1 * np.array(vector)
            relations_list.append(relation.tolist())
            node_list.append(node)
    prediction_list = get_probabilities(node_list=node_list, relations_list=relations_list, model=saved_model, k=k)
    print("The %d highest %s predictions for %s" % (k, entity_type, entity))
    return prediction_list

def find_chemicals(*, entity, embeddings, graph=None, node_mapping):
    relations_list=[]
    node_list = []
    node_id = node_mapping.loc[node_mapping["NodeName"] == entity, "NodeID"].iloc[0]
    node1 = embeddings[str(node_id)]
    for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
        if node == entity:
            continue
        if graph != None:
            if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                continue
        namespace = node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeNamespace"].iloc[0]
        if namespace != 'pubchem':
            continue
        relation = node1 * np.array(vector)
        relations_list.append(relation.tolist())
        node_list.append(node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeName"].iloc[0])
    return node_list, relations_list

def find_targets(*, entity, embeddings, graph=None, node_mapping):
    relations_list=[]
    node_list = []
    node_id = node_mapping.loc[node_mapping["NodeName"] == entity, "NodeID"].iloc[0]
    node1 = embeddings[str(node_id)]
    for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
        if node == entity:
            continue
        if graph != None:
            if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                continue
        namespace = node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeNamespace"].iloc[0]
        if namespace != 'uniprot':
            continue
        relation = node1 * np.array(vector)
        relations_list.append(relation.tolist())
        node_list.append(node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeName"].iloc[0])
    return node_list, relations_list

def find_phenotypes(*, entity, embeddings, graph=None, node_mapping):
    relations_list=[]
    node_list = []
    node_id = node_mapping.loc[node_mapping["NodeName"] == entity, "NodeID"].iloc[0]
    node1 = embeddings[str(node_id)]
    for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
        if node == entity:
            continue
        if graph != None:
            if graph.has_edge(entity, node) or graph.has_edge(node, entity):
                continue
        namespace = node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeNamespace"].iloc[0]
        if namespace != 'umls':
            continue
        relation = node1 * np.array(vector)
        relations_list.append(relation.tolist())
        node_list.append(node_mapping.loc[node_mapping["NodeID"] == int(node), "NodeName"].iloc[0])
    return node_list, relations_list

def get_probabilities(*, node_list, relations_list, model, k):
    all_prob = {}
    prob_list = model.predict_proba(relations_list)[:,1]
    for i in range(len(node_list)):
        all_prob[node_list[i]] = prob_list[i]
    sorted_list = sorted(all_prob.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_list[:k]

