# -*- coding: utf-8 -*-

"""

Find new relations between entities.

This file contains functions that find predicted relations from a logistic regression model given model embeddings
The model and embeddings are trained and created from a graph containing drugs, targets and side effects.
The graph used contained nodeIDs that can be mapped using a tsv file

"""

from operator import itemgetter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from tqdm import tqdm

from bionev.utils import load_embedding


class Predictor:
    def __init__(self, *, model, mapping, embeddings, graph=None):
        self.model = model
        self.mapping = mapping
        self.embeddings = embeddings
        self.graph = graph

    @classmethod
    def from_paths(cls, *, model_path, embeddings_path, mapping_path, graph_path=None):
        embeddings = load_embedding(embeddings_path)
        model = joblib.load(model_path)
        if graph_path is not None:
            graph = nx.read_edgelist(graph_path)
        else:
            graph = None
        node_mapping = pd.read_csv(mapping_path, sep='\t')
        return Predictor(model=model, mapping=node_mapping, graph=graph, embeddings=embeddings)

    def find_new_relations(
            self,
            entity_name=None,
            entity_identifier=None,
            entity_type=None,
            k: int = 30,
    ):
        return find_new_relations(
            entity_name=entity_name,
            entity_identifier=entity_identifier,
            entity_type=entity_type,
            k=k,
            saved_model=self.model,
            node_mapping=self.mapping,
            embeddings=self.embeddings,
            graph=self.graph
        )

    def find_new_relation(
            self,
            *,
            node_id_1=None,
            node_id_2=None,
            entity_name_1=None,
            entity_name_2=None,
            entity_id_1=None,
            entity_id_2=None,
    ) -> dict:
        if node_id_1 is not None:
            pass
        elif entity_name_1 is not None:
            node_id_1 = self.mapping.loc[self.mapping["name"] == entity_name_1, "node_id"].iloc[0]
        elif entity_id_1 is not None:
            node_id_1 = self.mapping.loc[self.mapping["identifier"] == entity_id_1, "node_id"].iloc[0]
        else:
            raise Exception("You need to provide information about the entity (node_id, entity_id or entity_name)")
        node1 = np.array(self.embeddings[str(node_id_1)])
        if node_id_2 is not None:
            pass
        elif entity_name_2 is not None:
            node_id_2 = self.mapping.loc[self.mapping["name"] == entity_name_2, "node_id"].iloc[0]
        elif entity_id_2 is not None:
            node_id_2 = self.mapping.loc[self.mapping["identifier"] == entity_id_2, "node_id"].iloc[0]
        else:
            raise Exception("You need to provide information about the entity (node_id, entity_id or entity_name)")
        node2 = np.array(self.embeddings[str(node_id_2)])
        x1 = node1 * node2
        x = [x1.tolist()]
        return {
            'entity_1':
                {
                    'node_id': node_id_1,
                    'namespace': self.mapping.loc[self.mapping["node_id"] == int(node_id_1), "namespace"].iloc[0],
                    'name': self.mapping.loc[self.mapping["node_id"] == int(node_id_1), "name"].iloc[0],
                    'identifier': self.mapping.loc[self.mapping["node_id"] == int(node_id_1), "identifier"].iloc[0]
                },
            'entity_2':
                {
                    'node_id': node_id_2,
                    'namespace': self.mapping.loc[self.mapping["node_id"] == int(node_id_2), "namespace"].iloc[0],
                    'name': self.mapping.loc[self.mapping["node_id"] == int(node_id_2), "name"].iloc[0],
                    'identifier': self.mapping.loc[self.mapping["node_id"] == int(node_id_2), "identifier"].iloc[0]
                },
            'probability': self.model.predict_proba(x)[:, 1][0]
        }


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
    return {
        'query': {
            'entity': entity_info,
            'k': k,
            'type': entity_type,
        },
        'predictions': prediction_list,
    }


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
        node_info = {
            'node_id': int(node),
            'namespace': namespace,
            'identifier': node_mapping.loc[node_mapping["node_id"] == int(node), "identifier"].iloc[0],
            'name': node_mapping.loc[node_mapping["node_id"] == int(node), "name"].iloc[0]
        }
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
        node_info = {
            'node_id': int(node),
            'namespace': namespace,
            'identifier': node_mapping.loc[node_mapping["node_id"] == int(node), "identifier"].iloc[0],
            'name': node_mapping.loc[node_mapping["node_id"] == int(node), "name"].iloc[0]
        }
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
        node_info = {
            'node_id': int(node),
            'namespace': namespace,
            'identifier': node_mapping.loc[node_mapping["node_id"] == int(node), "identifier"].iloc[0],
            'name': node_mapping.loc[node_mapping["node_id"] == int(node), "name"].iloc[0]
        }
        node_list.append(node_info)
    return node_list, relations_list


def get_probabilities(*, node_list, relations_list, model, k=None):
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
    prob_list = model.predict_proba(relations_list)[:, 1]
    all_prob = [
        {
            'probability': prob,
            **node
        }
        for node, prob in zip(node_list, prob_list)
    ]
    sorted_list = sorted(all_prob, key=itemgetter('probability'), reverse=True)
    if k is not None:
        return sorted_list[:k]
    else:
        return sorted_list
