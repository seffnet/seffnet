# -*- coding: utf-8 -*-

"""Utilities for :mod:`seffnet`."""

import json
import os
import random
from itertools import chain
from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import optuna
import pandas as pd
import pybel
import seaborn as sns
from bionev.pipeline import split_train_test_graph, train_test_graph
from bionev.utils import read_graph

from .constants import (
    DEFAULT_MAPPING_PATH, PUBCHEM_NAMESPACE, RESULTS_TYPE_TO_DSL, RESULTS_TYPE_TO_NAMESPACE,
    UNIPROT_NAMESPACE,
)


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
    input_graph = read_graph(input_path, weighted=weighted)
    if training_path and testing_path is not None:
        graph_train, testing_pos_edges, train_graph_filename = train_test_graph(
            training_path,
            testing_path,
            weighted=weighted,
        )
    else:
        graph_train, testing_pos_edges, train_graph_filename = split_train_test_graph(
            input_graph=input_path,
            weighted=weighted,
        )
    return input_graph, graph_train, testing_pos_edges, train_graph_filename


def create_subgraph(  # noqa: C901
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

    # Invert the weights
    for source, target in fullgraph.edges():
        for key, edge_data in fullgraph[source][target].items():
            fullgraph[source][target][key]['weight'] = 1 - edge_data['weight']

    mapping_df = pd.read_csv(
        mapping_path,
        sep="\t",
        dtype={'identifier': str},
        index_col=False,
    ).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        if row['namespace'] != PUBCHEM_NAMESPACE:
            continue
        if row['name'] is None:
            continue

        _k = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=row['identifier'])
        _v = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, name=row['name'])
        mapping_dict[_k] = _v

    source_dsl = RESULTS_TYPE_TO_DSL[source_type]
    source_namespace = RESULTS_TYPE_TO_NAMESPACE[source_type]
    if source_type == 'chemical':
        source = source_dsl(namespace=source_namespace, identifier=source_identifier)  # FIXME why no source_name?
    elif source_type == 'target':
        source = source_dsl(namespace=source_namespace, identifier=source_identifier, name=source_name)
    elif source_type == 'phenotype':
        source = source_dsl(namespace=source_namespace, identifier=source_identifier, name=source_name)
    else:
        raise KeyError

    target_dsl = RESULTS_TYPE_TO_DSL[target_type]
    target_namespace = RESULTS_TYPE_TO_NAMESPACE[target_type]
    if target_type == 'chemical':
        target = target_dsl(namespace=target_namespace, identifier=target_identifier)  # FIXME why no target_name?
    elif target_type == 'target':
        target = target_dsl(namespace=target_namespace, identifier=target_identifier, name=target_name)
    elif target_type == 'phenotype':
        target = target_dsl(namespace=target_namespace, identifier=target_identifier, name=target_name)
    else:
        raise KeyError

    fullgraph_undirected = fullgraph.to_undirected()
    paths = [
        path
        for path in nx.all_shortest_paths(
            fullgraph_undirected,
            source=source,
            target=target,
            weight='weight' if weighted else None,
        )
    ]

    if len(paths) > 100:
        paths = random.sample(paths, 10)

    subgraph_nodes = {
        node
        for path in paths
        for node in path
    }

    if common_targets:
        for neighbor in chain(fullgraph.neighbors(source), fullgraph.neighbors(target)):
            if neighbor.namespace == UNIPROT_NAMESPACE:
                subgraph_nodes.add(neighbor)

    subgraph = fullgraph.subgraph(subgraph_nodes).copy()
    # the subgraph has the same number of degrees from the fullgraph so we need to create a copy of it
    digraph = nx.DiGraph(subgraph)  # FIXME are the digraph and subgraph variables mixed up here?
    if common_targets:
        subgraph.remove_nodes_from([node for node in digraph if digraph.degree(node) < 2])
    subgraph = nx.relabel_nodes(subgraph, mapping_dict)
    return subgraph


def get_boxplot(
    *,
    dir_path: str,
    metric: str = 'mcc',
):
    """Make a boxplot from repeat output.

    :param dir_path: the path with the outputs from the repeats.
    :param metric: the type of metric to plot in the boxplot.
    :return: boxplot
    """
    method = []
    metric_list = []

    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename)) as file:
            data = json.load(file)
        for v in data.values():
            if v['input'] == 'node_shuffle':
                method.append('shuffle')
                metric_list.append(v['results'][metric])
            elif v['input'] == 'random':
                method.append('random')
                metric_list.append(v['results'][metric])
            else:
                method.append(v['method'])
                metric_list.append(v['results'][metric])
    df = pd.DataFrame(list(zip(method, metric_list)), columns=['method', metric])
    df = df.sort_values(by=[metric], ascending=False)
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x="method", y=metric, data=df)
    boxplot.set_title('Robustness Analysis', fontdict={'fontsize':36})
    boxplot.set_xlabel('')
    boxplot.set_ylabel('MCC', fontdict={'fontsize':24})
    return df, boxplot
