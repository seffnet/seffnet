# -*- coding: utf-8 -*-

"""Utilities for :mod:`seffnet`."""


import json
import os
import random
from typing import Any, Mapping

import networkx as nx
import optuna
import pandas as pd
import pybel
import seaborn as sns
from bionev.pipeline import train_test_graph, split_train_test_graph
from bionev.utils import read_graph

from .constants import DEFAULT_MAPPING_PATH


def study_to_json(study: optuna.Study, prediction_task) -> Mapping[str, Any]:
    """Serialize a study to JSON."""
    if prediction_task == 'link_prediction':
        return {
            'n_trials': len(study.trials),
            'name': study.study_name,
            'id': study.study_id,
            'prediction_task': prediction_task,
            'start': study.user_attrs['Date'],
            'seed': study.user_attrs['Seed'],
            'best': {
                'mcc': study.best_trial.user_attrs['mcc'],
                'accuracy': study.best_trial.user_attrs['accuracy'],
                'auc_roc': study.best_trial.user_attrs['auc_roc'],
                'auc_pr': study.best_trial.user_attrs['auc_pr'],
                'f1': study.best_trial.user_attrs['f1'],
                'method': study.best_trial.user_attrs['method'],
                'classifier': study.best_trial.user_attrs['classifier'],
                'inner_seed': study.best_trial.user_attrs['inner_seed'],
                'params': study.best_params,
                'trial': study.best_trial.number,
                'value': study.best_value,
            },
        }
    else:
        return {
            'n_trials': len(study.trials),
            'name': study.study_name,
            'id': study.study_id,
            'prediction_task': prediction_task,
            'start': study.user_attrs['Date'],
            'seed': study.user_attrs['Seed'],
            'best': {
                'accuracy': study.best_trial.user_attrs['accuracy'],
                'micro_f1': study.best_trial.user_attrs['micro_f1'],
                'macro_f1': study.best_trial.user_attrs['macro_f1'],
                'method': study.best_trial.user_attrs['method'],
                'classifier': study.best_trial.user_attrs['classifier'],
                'inner_seed': study.best_trial.user_attrs['inner_seed'],
                'params': study.best_params,
                'trial': study.best_trial.number,
                'value': study.best_value,
            },
        }


def create_graphs(*, input_path, training_path, testing_path, weighted):
    """Create the training/testing graphs needed for evalution."""
    input_graph = read_graph(input, weighted=weighted)
    if training_path and testing_path is not None:
        graph_train, testing_pos_edges, train_graph_filename = train_test_graph(
            training_path,
            testing_path,
            weighted=weighted,
        )
    else:
        graph_train, testing_pos_edges, train_graph_filename = split_train_test_graph(
            input_edgelist=input_path,
            weighted=weighted,
        )
    return input_graph, graph_train, testing_pos_edges, train_graph_filename


def create_subgraph(
    *,
    fullgraph_path,
    source_name=None,
    source_identifier=None,
    source_type,
    target_name=None,
    target_identifier=None,
    target_type,
    weighted=False,
    mapping_path=DEFAULT_MAPPING_PATH,
    common_targets: bool = False,
):
    """Create subgraph."""
    fullgraph = pybel.from_pickle(fullgraph_path)
    for edge in fullgraph.edges():
        for iden, edge_d in fullgraph[edge[0]][edge[1]].items():
            fullgraph[edge[0]][edge[1]][iden]['weight'] = 1 - edge_d['weight']
    mapping_df = pd.read_csv(
        mapping_path,
        sep="\t",
        dtype={'identifier': str},
        index_col=False,
    ).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    mapping_dict = {}
    for ind, row in mapping_df.iterrows():
        if row['namespace'] != 'pubchem.compound':
            continue
        if row['name'] is None:
            continue
        mapping_dict[
            pybel.dsl.Abundance(namespace='pubchem.compound', identifier=row['identifier'])] = pybel.dsl.Abundance(
            namespace='pubchem.compound', name=row['name'])
    if source_type == 'chemical':
        source = pybel.dsl.Abundance(namespace='pubchem.compound', identifier=source_identifier)
    elif source_type == 'protein':
        source = pybel.dsl.Protein(namespace='uniprot', name=source_name, identifier=source_identifier)
    elif source_type == 'phenotype':
        source = pybel.dsl.Pathology(namespace='umls', name=source_name, identifier=source_identifier)
    else:
        raise Exception('Source type is not valid!')
    if target_type == 'chemical':
        target = pybel.dsl.Abundance(namespace='pubchem.compound', identifier=target_identifier)
    elif target_type == 'protein':
        target = pybel.dsl.Protein(namespace='uniprot', name=target_name, identifier=target_identifier)
    elif target_type == 'phenotype':
        target = pybel.dsl.Pathology(namespace='umls', name=target_name, identifier=target_identifier)
    else:
        raise Exception('Target type is not valid!')
    fullgraph_undirected = fullgraph.to_undirected()
    if weighted:
        paths = [p for p in nx.all_shortest_paths(fullgraph_undirected, source=source, target=target, weight='weight')]
    else:
        paths = [p for p in nx.all_shortest_paths(fullgraph_undirected, source=source, target=target)]
    subgraph_nodes = []
    if len(paths) > 100:
        for path in random.sample(paths, 10):
            for node in path:
                if node in subgraph_nodes:
                    continue
                subgraph_nodes.append(node)
    else:
        for path in paths:
            for node in path:
                if node in subgraph_nodes:
                    continue
                subgraph_nodes.append(node)
    if common_targets:
        for neighbor in fullgraph.neighbors(source):
            if neighbor.namespace != 'uniprot':
                continue
            subgraph_nodes.append(neighbor)
        for neighbor in fullgraph.neighbors(target):
            if neighbor.namespace != 'uniprot':
                continue
            subgraph_nodes.append(neighbor)
    subgraph = fullgraph.subgraph(subgraph_nodes).copy()
    # the subgraph has the same number of degrees from the fullgraph so we need to create a copy of it
    digraph = nx.DiGraph(subgraph)
    if common_targets:
        remove = [node for node in digraph.nodes() if digraph.degree(node) < 2]
        subgraph.remove_nodes_from(remove)
    subgraph = nx.relabel_nodes(subgraph, mapping_dict)
    return subgraph


def get_boxplot(
        *,
        dir_path: str,
        metric: str = 'mcc',
):
    """
    Make boxplot from repeat output.

    :param dir_path: the path with the outputs from the repeats.
    :param metric: the type of metric to plot in the boxplot.
    :return: boxplot
    """
    method = []
    metric_list = []

    for filename in os.listdir(dir_path):
        with open(dir_path+filename) as json_file:
            data = json.load(json_file)
            for key in data.keys():
                method.append(data[key]['method'])
                metric_list.append(data[key]['results'][metric])
    df = pd.DataFrame(list(zip(method, metric_list)), columns=['method', metric])
    boxplot = sns.boxplot(x="method", y=metric, data=df)
    return boxplot
