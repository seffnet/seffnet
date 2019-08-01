# -*- coding: utf-8 -*-

"""

Find new relations between entities.

This file contains functions that find predicted relations from a logistic regression model given model embeddings
The model and embeddings are trained and created from a graph containing drugs, targets and side effects.
The graph used contained nodeIDs that can be mapped using a tsv file

"""

import numpy as np
from tqdm import tqdm


def find_new_relations(
        *,
        entity_name=None,
        entity_identifier=None,
        saved_model,
        node_mapping,
        embeddings,
        graph=None,
        entity_type=None,
        k: int = 30,
):
    """
    Find new relations to specific entity.

    Get all the relations of specific entity_type (if chosen) or all types (if None).
    Finds their probabilities from the saved_model, and return the top k predictions.

    :param entity_name: the entity we want to find predictions with
    :param entity_identifier: the identifier of the entity we want to find predictions with
    :param saved_model: the log regression model created from the graph
    :param node_mapping: a dataframe containing the original names of the nodes mapped to their IDs on the graph
    :param embeddings: the embeddings created from the graph
    :param graph: the graph that was used to train the model
    :param entity_type: can be phenotype, chemical or target
    :param k: the amount of relations we want to find for the entity
    :return: a list of tuples containing the predicted entities and their probabilities
    """
    if entity_name is not None:
        entity_id = node_mapping.loc[node_mapping["name"] == entity_name, "node_id"].iloc[0]
    elif entity_identifier is not None:
        entity_id = node_mapping.loc[node_mapping["identifier"] == entity_identifier, "node_id"].iloc[0]
    else:
        return 'No input entity for prediction'
    entity_info = {
        'node_id': int(entity_id),
        'namespace': node_mapping.loc[node_mapping["node_id"] == int(entity_id), "namespace"].iloc[0],
        'identifier': node_mapping.loc[node_mapping["node_id"] == int(entity_id), "identifier"].iloc[0],
        'name': node_mapping.loc[node_mapping["node_id"] == int(entity_id), "name"].iloc[0]
    }

    entity_vector = embeddings[str(entity_id)]
    if entity_type == 'chemical':
        node_list, relations_list = find_chemicals(
            entity_vector=entity_vector,
            entity_id=entity_id,
            embeddings=embeddings,
            node_mapping=node_mapping,
            graph=graph
        )
    elif entity_type == 'phenotype':
        node_list, relations_list = find_phenotypes(
            entity_vector=entity_vector,
            entity_id=entity_id,
            embeddings=embeddings,
            node_mapping=node_mapping,
            graph=graph
        )
    elif entity_type == 'target':
        node_list, relations_list = find_targets(
            entity_vector=entity_vector,
            entity_id=entity_id,
            embeddings=embeddings,
            node_mapping=node_mapping,
            graph=graph
        )
    else:
        relations_list = []
        node_list = []
        for node_2, vector in tqdm(embeddings.items(), desc="creating relations list"):
            if node_2 == entity_id:
                continue
            if graph is not None:
                if graph.has_edge(entity_id, node_2) or graph.has_edge(node_2, entity_id):
                    continue
            relation = entity_vector * np.array(vector)
            relations_list.append(relation.tolist())
            node_info = {
                'node_id': int(node_2),
                'namespace': node_mapping.loc[node_mapping["node_id"] == int(node_2), "namespace"].iloc[0],
                'identifier': node_mapping.loc[node_mapping["node_id"] == int(node_2), "identifier"].iloc[0],
                'name': node_mapping.loc[node_mapping["node_id"] == int(node_2), "name"].iloc[0]
            }
            node_list.append(node_info)
    prediction_list = get_probabilities(node_list=node_list, relations_list=relations_list, model=saved_model, k=k)
    print("The %d highest %s predictions for %s" % (k, entity_type, entity_info))
    return prediction_list


def find_chemicals(*, entity_id, entity_vector, embeddings, graph=None, node_mapping):
    """
    Find all relations of the entity with chemical entities only.

    :param entity_id: the entity we want to find predictions with
    :param entity_vector: the vector of the entity
    :param embeddings: the embeddings created from the graph
    :param graph: the graph that was used to train the model
    :param node_mapping: a dataframe containing the original names of the nodes mapped to their IDs on the graph
    :return node_list: a list of the nodes with relations,
    relations_list: the edge embedding of the two nodes in the relation
    """
    relations_list = []
    node_list = []
    for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
        node_info = {}
        if node == entity_id:
            continue
        if graph is not None:
            if graph.has_edge(entity_id, node) or graph.has_edge(node, entity_id):
                continue
        namespace = node_mapping.loc[node_mapping["node_id"] == int(node), "namespace"].iloc[0]
        if namespace != 'pubchem':
            continue
        relation = entity_vector * np.array(vector)
        relations_list.append(relation.tolist())
        node_info['node_id'] = int(node)
        node_info['namespace'] = namespace
        node_info['identifier'] = node_mapping.loc[node_mapping["node_id"] == int(node), "identifier"].iloc[0]
        node_info['name'] = node_mapping.loc[node_mapping["node_id"] == int(node), "name"].iloc[0]
        node_list.append(node_info)
    return node_list, relations_list


def find_targets(*, entity_vector, entity_id, embeddings, graph=None, node_mapping):
    """
    Find all relations of the entity with protein entities only.

    :param entity_id: the entity we want to find predictions with
    :param entity_vector: the vector of the entity
    :param embeddings: the embeddings created from the graph
    :param graph: the graph that was used to train the model
    :param node_mapping: a dataframe containing the original names of the nodes mapped to their IDs on the graph
    :return node_list: a list of the nodes with relations,
    relations_list: the edge embedding of the two nodes in the relation
    """
    relations_list = []
    node_list = []
    for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
        node_info = {}
        if node == entity_id:
            continue
        if graph is not None:
            if graph.has_edge(entity_id, node) or graph.has_edge(node, entity_id):
                continue
        namespace = node_mapping.loc[node_mapping["node_id"] == int(node), "namespace"].iloc[0]
        if namespace != 'uniprot':
            continue
        relation = entity_vector * np.array(vector)
        relations_list.append(relation.tolist())
        node_info['node_id'] = int(node)
        node_info['namespace'] = namespace
        node_info['identifier'] = node_mapping.loc[node_mapping["node_id"] == int(node), "identifier"].iloc[0]
        node_info['name'] = node_mapping.loc[node_mapping["node_id"] == int(node), "name"].iloc[0]
        node_list.append(node_info)
    return node_list, relations_list


def find_phenotypes(*, entity_vector, entity_id, embeddings, graph=None, node_mapping):
    """
    Find all relations of the entity with phenotype entities only.

    :param entity_id: the entity we want to find predictions with
    :param entity_vector: the vector of the entity
    :param embeddings: the embeddings created from the graph
    :param graph: the graph that was used to train the model
    :param node_mapping: a dataframe containing the original names of the nodes mapped to their IDs on the graph
    :return node_list: a list of the nodes with relations,
    relations_list: the edge embedding of the two nodes in the relation
    """
    relations_list = []
    node_list = []
    for node, vector in tqdm(embeddings.items(), desc="creating relations list"):
        node_info = {}
        if node == entity_id:
            continue
        if graph is not None:
            if graph.has_edge(entity_id, node) or graph.has_edge(node, entity_id):
                continue
        namespace = node_mapping.loc[node_mapping["node_id"] == int(node), "namespace"].iloc[0]
        if namespace != 'umls':
            continue
        relation = entity_vector * np.array(vector)
        relations_list.append(relation.tolist())
        node_info['node_id'] = int(node)
        node_info['namespace'] = namespace
        node_info['identifier'] = node_mapping.loc[node_mapping["node_id"] == int(node), "identifier"].iloc[0]
        node_info['name'] = node_mapping.loc[node_mapping["node_id"] == int(node), "name"].iloc[0]
        node_list.append(node_info)
    return node_list, relations_list


def get_probabilities(*, node_list, relations_list, model, k):
    """
    Get probabilities from log model.

    Get the probabilities of all the relations in the list from the log model.
    Also sort the found probabilities by highest to lowest, then return the k highest probabilities.

    :param node_list: the list of the nodes with relations to the entity
    :param relations_list: the list of edge embedding of the two nodes
    :param model: the logistic regression model
    :param k: the number of relations to be output
    :return sorted_list[:k]: the k first probabilities in the list, type= list of tuples
    """
    all_prob = []
    output_dict = {}
    prob_list = model.predict_proba(relations_list)[:, 1]
    for i in range(len(node_list)):
        all_prob.append((node_list[i], prob_list[i]))
    sorted_list = sorted(all_prob, key=lambda kv: kv[1], reverse=True)
    i = 1
    for tup in sorted_list[:k]:
        tup[0]['probability'] = tup[1]
        output_dict[i] = tup[0]
        i += 1
    return output_dict
