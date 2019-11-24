# -*- coding: utf-8 -*-

"""Pipeline for :mod:`seffnet`."""

import datetime
import getpass
import json
import random
from typing import Optional

import networkx as nx
import numpy as np
from bionev.embed_train import embedding_training
from bionev.pipeline import create_prediction_model, do_link_prediction, do_node_classification
from bionev.utils import read_node_labels, read_graph, split_train_test_graph
from tqdm import tqdm
import xswap

from .optimization import (
    deepwalk_optimization, grarep_optimization, hope_optimization, line_optimization,
    node2vec_optimization, sdne_optimization,
)
from .utils import create_graphs, study_to_json


def do_evaluation(
    *,
    input_path,
    training_path: Optional[str] = None,
    testing_path: Optional[str] = None,
    method,
    prediction_task,
    dimensions: int = 300,
    number_walks: int = 8,
    walk_length: int = 8,
    window_size: int = 4,
    p: float = 1.5,
    q: float = 2.1,
    alpha: float = 0.1,
    beta: float = 4,
    epochs: int = 5,
    kstep: int = 4,
    order: int = 3,
    embeddings_path: Optional[str] = None,
    predictive_model_path: Optional[str] = None,
    training_model_path: Optional[str] = None,
    evaluation_file: Optional[str] = None,
    classifier_type: Optional[str] = None,
    weighted: bool = False,
    labels_file: Optional[str] = None,
):
    """Train and evaluate an NRL model."""
    if prediction_task == 'link_prediction':
        node_list = None
        labels = None
        graph, graph_train, testing_pos_edges, train_graph_filename = create_graphs(
            input_path=input_path,
            training_path=training_path,
            testing_path=testing_path,
            weighted=weighted,
        )
    else:
        if not labels_file:
            raise ValueError("No input label file. Exit.")
        node_list, labels = read_node_labels(labels_file)
        train_graph_filename = input_path
        graph, graph_train, testing_pos_edges = None, None, None

    model = embedding_training(
        train_graph_filename=train_graph_filename,
        method=method,
        dimensions=dimensions,
        number_walks=number_walks,
        walk_length=walk_length,
        window_size=window_size,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        kstep=kstep,
        order=order,
        weighted=weighted,
    )
    if training_model_path is not None:
        model.save_model(training_model_path)
    if embeddings_path is not None:
        model.save_embeddings(embeddings_path)
    if method == 'LINE':
        embeddings = model.get_embeddings_train()
    else:
        embeddings = model.get_embeddings()

    _results = dict(
        input=input_path,
        method=method,
        dimension=dimensions,
        user=getpass.getuser(),
        date=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'),
    )
    if prediction_task == 'link_prediction':
        auc_roc, auc_pr, accuracy, f1, mcc = do_link_prediction(
            embeddings=embeddings,
            original_graph=graph,
            train_graph=graph_train,
            test_pos_edges=testing_pos_edges,
            save_model=predictive_model_path,
            classifier_type=classifier_type,
        )
        _results['results'] = dict(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            accuracy=accuracy,
            f1=f1,
            mcc=mcc,
        )
    else:
        accuracy, macro_f1, micro_f1, mcc = do_node_classification(
            embeddings=embeddings,
            node_list=node_list,
            labels=labels,
            save_model=predictive_model_path,
            classifier_type=classifier_type,
        )
        _results['results'] = dict(
            accuracy=accuracy,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            mcc=mcc,
        )
    if evaluation_file is not None:
        json.dump(_results, evaluation_file, sort_keys=True, indent=2)
    return _results


def do_optimization(
    *,
    method,
    input_path,
    training_path,
    testing_path,
    trials,
    dimensions_range,
    storage,
    name,
    output,
    prediction_task,
    labels_file,
    classifier_type,
    study_seed,
    weighted: bool = False,
):
    """Run optimization a specific method and graph."""
    np.random.seed(study_seed)
    random.seed(study_seed)

    if prediction_task == 'link_prediction':
        node_list, labels = None, None
        graph, graph_train, testing_pos_edges, train_graph_filename = create_graphs(
            input_path=input_path,
            training_path=training_path,
            testing_path=testing_path,
            weighted=weighted,
        )
    elif not labels_file:
        raise ValueError("No input label file. Exit.")
    else:
        node_list, labels = read_node_labels(labels_file)
        graph, graph_train, testing_pos_edges, train_graph_filename = None, None, None, input_path

    if method == 'HOPE':
        study = hope_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
            prediction_task=prediction_task,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            weighted=weighted,
            seed=study_seed,
        )

    elif method == 'DeepWalk':
        study = deepwalk_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            study_seed=study_seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
            prediction_task=prediction_task,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            weighted=weighted,
        )

    elif method == 'node2vec':
        study = node2vec_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            study_seed=study_seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
            prediction_task=prediction_task,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            weighted=weighted,
        )

    elif method == 'GraRep':
        study = grarep_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            study_seed=study_seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
            prediction_task=prediction_task,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            weighted=weighted,
        )

    elif method == 'SDNE':
        study = sdne_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            study_seed=study_seed,
            storage=storage,
            study_name=name,
            prediction_task=prediction_task,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            weighted=weighted,
        )

    else:
        study = line_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            study_seed=study_seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
            prediction_task=prediction_task,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            weighted=weighted,
        )

    study_json = study_to_json(study, prediction_task)
    json.dump(study_json, output, indent=2, sort_keys=True)


def train_model(
    *,
    input_path,
    method,
    dimensions: int = 300,
    number_walks: int = 8,
    walk_length: int = 8,
    window_size: int = 4,
    p: float = 1.5,
    q: float = 2.1,
    alpha: float = 0.1,
    beta: float = 4,
    epochs: int = 5,
    kstep: int = 4,
    order: int = 3,
    embeddings_path: Optional[str] = None,
    predictive_model_path: Optional[str] = None,
    training_model_path: Optional[str] = None,
    classifier_type: Optional[str] = None,
    weighted: bool = False,
    labels_file: Optional[str] = None,
    prediction_task,
):
    """Train a graph with an NRL model."""
    node_list, labels = None, None
    if prediction_task == 'node_classification':
        if not labels_file:
            raise ValueError("No input label file. Exit.")
        node_list, labels = read_node_labels(labels_file)
    model = embedding_training(
        train_graph_filename=input_path,
        method=method,
        dimensions=dimensions,
        number_walks=number_walks,
        walk_length=walk_length,
        window_size=window_size,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        kstep=kstep,
        order=order,
        weighted=weighted,
    )
    if training_model_path is not None:
        model.save_model(training_model_path)
    model.save_embeddings(embeddings_path)
    original_graph = nx.read_edgelist(input_path)
    if method == 'LINE':
        embeddings = model.get_embeddings_train()
    else:
        embeddings = model.get_embeddings()
    if prediction_task == 'link_prediction':
        create_prediction_model(
            embeddings=embeddings,
            original_graph=original_graph,
            save_model=predictive_model_path,
            classifier_type=classifier_type,
        )
    else:
        do_node_classification(
            embeddings=embeddings,
            node_list=node_list,
            labels=labels,
            classifier_type=classifier_type,
            save_model=predictive_model_path,
        )


def repeat_experiment(
    *,
    input_path,
    training_path=None,
    testing_path=None,
    method,
    dimensions=300,
    number_walks=8,
    walk_length=8,
    window_size=4,
    p=1.5,
    q=2.1,
    alpha=0.1,
    beta=4,
    epochs=5,
    kstep=4,
    order=3,
    n=10,
    evaluation_file=None,
    weighted: bool = False,
    prediction_task,
    randomization=None,
):
    """Repeat an experiment several times."""
    if randomization is None:
        all_results = {
            i: do_evaluation(
                input_path=input_path,
                training_path=training_path,
                testing_path=testing_path,
                method=method,
                dimensions=dimensions,
                number_walks=number_walks,
                walk_length=walk_length,
                window_size=window_size,
                p=p,
                q=q,
                alpha=alpha,
                beta=beta,
                epochs=epochs,
                kstep=kstep,
                order=order,
                embeddings_path=None,
                predictive_model_path=None,
                evaluation_file=None,
                weighted=weighted,
                prediction_task=prediction_task,
            )
            for i in tqdm(range(n), desc="Repeating experiment")
        }
    else:
        graph = read_graph(input_path, weighted=weighted)
        all_results = {
            i: randomize(
                randomization_method=randomization,
                input_graph=graph,
                method=method,
                dimensions=dimensions,
                number_walks=number_walks,
                walk_length=walk_length,
                window_size=window_size,
                p=p,
                q=q,
                alpha=alpha,
                beta=beta,
                epochs=epochs,
                kstep=kstep,
                order=order,
                weighted=weighted,
            )
            for i in tqdm(range(n), desc="Repeating randomization experiment")
        }
    if evaluation_file is not None:
        json.dump(all_results, evaluation_file, sort_keys=True, indent=2)
    return all_results


def randomize(
    *,
    randomization_method,
    input_graph,
    method,
    dimensions: int = 300,
    number_walks: int = 8,
    walk_length: int = 8,
    window_size: int = 4,
    p: float = 1.5,
    q: float = 2.1,
    alpha: float = 0.1,
    beta: float = 4,
    epochs: int = 5,
    kstep: int = 4,
    order: int = 3,
    weighted: bool = False,
):
    if randomization_method == 'xswap':
        edges = list(input_graph.edges())
        permuted_edges, permutation_statistics = xswap.permute_edge_list(
            edges, allow_self_loops=True, allow_antiparallel=True,
            multiplier=10)
        random_graph = nx.add_edges_from(permuted_edges)
    elif randomization_method == 'random':
        random_graph = nx.gnm_random_graph(len(input_graph.nodes()), len(input_graph.edges()))
    elif randomization_method == 'powerlaw':
        random_graph = nx.powerlaw_cluster_graph(len(input_graph.nodes()), len(input_graph.edges()))
    else:
        return "Randomization method not valid."
    _, graph_train, testing_pos_edges, train_graph_filename = split_train_test_graph(
        input_graph=random_graph,
        weighted=weighted,
    )
    model = embedding_training(
        train_graph_filename=train_graph_filename,
        method=method,
        dimensions=dimensions,
        number_walks=number_walks,
        walk_length=walk_length,
        window_size=window_size,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        kstep=kstep,
        order=order,
        weighted=weighted,
    )
    if method == 'LINE':
        embeddings = model.get_embeddings_train()
    else:
        embeddings = model.get_embeddings()

    _results = dict(
        input=input_graph,
        method=method,
        dimension=dimensions,
        user=getpass.getuser(),
        date=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'),
    )
    auc_roc, auc_pr, accuracy, f1, mcc = do_link_prediction(
        embeddings=embeddings,
        original_graph=input_graph,
        train_graph=graph_train,
        test_pos_edges=testing_pos_edges,
        save_model=None,
        classifier_type=None,
    )
    _results['results'] = dict(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        accuracy=accuracy,
        f1=f1,
        mcc=mcc,
    )
    return _results
