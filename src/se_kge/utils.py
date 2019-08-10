# -*- coding: utf-8 -*-

"""Utilities for ``se_kge``."""
import datetime
import getpass
import json
from typing import Any, Mapping

import networkx as nx
import optuna
from bionev import pipeline
from bionev.embed_train import embedding_training

from .optimization import (
    deepwalk_optimization, grarep_optimization, hope_optimization, line_optimization,
    node2vec_optimization, sdne_optimization,
)


def study_to_json(study: optuna.Study) -> Mapping[str, Any]:
    """Serialize a study to JSON."""
    return {
        'n_trials': len(study.trials),
        'name': study.study_name,
        'id': study.study_id,
        'start': study.user_attrs['Date'],
        'best': {
            'mcc': study.best_trial.user_attrs['mcc'],
            'accuracy': study.best_trial.user_attrs['accuracy'],
            'auc_roc': study.best_trial.user_attrs['auc_roc'],
            'auc_pr': study.best_trial.user_attrs['auc_pr'],
            'f1': study.best_trial.user_attrs['f1'],
            'method': study.best_trial.user_attrs['method'],
            'params': study.best_params,
            'trial': study.best_trial.number,
            'value': study.best_value,
        },
    }


def create_graphs(*, input_path, training_path, testing_path, seed):
    if training_path and testing_path is not None:
        graph, graph_train, testing_pos_edges, train_graph_filename = pipeline.train_test_graph(
            input_path,
            training_path,
            testing_path,
        )
    else:
        graph, graph_train, testing_pos_edges, train_graph_filename = pipeline.split_train_test_graph(
            input_path,
            seed
        )
    return graph, graph_train, testing_pos_edges, train_graph_filename


def do_evaluation(
        *,
        input_path,
        training_path,
        testing_path,
        method,
        dimensions,
        number_walks,
        walk_length,
        window_size,
        p,
        q,
        alpha,
        beta,
        epochs,
        kstep,
        order,
        seed,
        embeddings_path,
        model_path,
        evaluation_file
):
    graph, graph_train, testing_pos_edges, train_graph_filename = create_graphs(
        input_path=input_path,
        training_path=training_path,
        testing_path=testing_path,
        seed=seed,
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
        seed=seed,
    )
    model.save_embeddings(embeddings_path)
    auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(
        embeddings=model.get_embeddings(),
        original_graph=graph,
        train_graph=graph_train,
        test_pos_edges=testing_pos_edges,
        seed=seed,
        save_model=model_path
    )
    _results = dict(
        input=input_path,
        method=method,
        dimension=dimensions,
        user=getpass.getuser(),
        date=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'),
        seed=seed,
    )
    _results['results'] = dict(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        accuracy=accuracy,
        f1=f1,
        mcc=mcc,
    )
    json.dump(_results, evaluation_file, sort_keys=True, indent=2)
    return _results


def do_optimization(
        *,
        method,
        input_path,
        training_path,
        testing_path,
        trials,
        seed,
        dimensions_range,
        storage,
        name,
        output,
):
    graph, graph_train, testing_pos_edges, train_graph_filename = create_graphs(
        input_path=input_path,
        training_path=training_path,
        testing_path=testing_path,
        seed=seed,
    )
    if method == 'HOPE':
        study = hope_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
        )

    elif method == 'DeepWalk':
        study = deepwalk_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
        )

    elif method == 'node2vec':
        study = node2vec_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
        )

    elif method == 'GraRep':
        study = grarep_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
        )

    elif method == 'SDNE':
        study = sdne_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            storage=storage,
            study_name=name,
        )

    else:
        study = line_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=dimensions_range,
            storage=storage,
            study_name=name,
        )

    study_json = study_to_json(study)
    json.dump(study_json, output, indent=2, sort_keys=True)


def train_model(
        *,
        input_path,
        method,
        dimensions,
        number_walks,
        walk_length,
        window_size,
        p,
        q,
        alpha,
        beta,
        epochs,
        kstep,
        order,
        seed,
        model_path,
        embeddings_path,
):
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
        seed=seed,
    )
    model.save_embeddings(embeddings_path)
    original_graph = nx.read_edgelist(input_path)
    pipeline.create_prediction_model(
        embeddings=model.get_embeddings(),
        original_graph=original_graph,
        seed=seed,
        save_model=model_path
    )
