# -*- coding: utf-8 -*-

"""Command line interface for ``se_kge``."""

import json
import random
import sys

import click
from bionev import pipeline

from .optimization import (
    deepwalk_optimization, grarep_optimization, hope_optimization, line_optimization,
    node2vec_optimization, sdne_optimization,
)
from .utils import study_to_json


@click.group()
def main():
    """Side Effects Knowledge Graph Embeddings."""


@main.command()
@click.option('--input-path', required=True, help='Input graph file. Only accepted edgelist format.')
@click.option('--training', default=None, help='training graph file. Only accepted edgelist format.')
@click.option('--testing', default=None, help='testing graph file. Only accepted edgelist format.')
@click.option('--method', required=True, help='The embedding learning method')
@click.option('--trials', default=50, type=int, help='the number of trials done to optimize hyperparameters')
@click.option('--dimensions-low', default=100, type=int, help='the range of dimensions to be optimized')
@click.option('--dimensions-high', default=300, type=int, help='the range of dimensions to be optimized')
@click.option('--seed', type=int, default=random.randint(1, 10000000))
@click.option('--storage', help="SQL connection string for study database. Example: sqlite:///optuna.db")
@click.option('--name', help="Name for the study")
@click.option('-o', '--output', type=click.File('w'), help="Output study summary", default=sys.stdout)
def optimize(
        input_path,
        training,
        testing,
        method,
        trials,
        dimensions_low,
        dimensions_high,
        seed,
        storage,
        name,
        output,
):
    """Run the optimization pipeline for a given method and graph."""
    if training and testing is not None:
        graph, graph_train, testing_pos_edges, train_graph_filename = pipeline.train_test_graph(
            input_path,
            training,
            testing,
        )
    else:
        graph, graph_train, testing_pos_edges, train_graph_filename = pipeline.split_train_test_graph(
            input_path,
            seed
        )
    if method == 'HOPE':
        study = hope_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=(dimensions_low, dimensions_high),
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
            dimensions_range=(dimensions_low, dimensions_high),
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
            dimensions_range=(dimensions_low, dimensions_high),
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
            dimensions_range=(dimensions_low, dimensions_high),
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

    elif method == 'LINE':
        study = line_optimization(
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=trials,
            seed=seed,
            dimensions_range=(dimensions_low, dimensions_high),
            storage=storage,
            study_name=name,
        )

    else:
        raise ValueError(f'Invalid method: {method}')

    study_json = study_to_json(study)
    json.dump(study_json, output, indent=2, sort_keys=True)


@main.command()
def train():
    """Train my model."""


@main.command()
@click.option('--host')
@click.option('--port', type=int)
def web(host, port):
    """Run the se_kge RESTful API."""
    from .web import get_app, api
    _app = get_app()
    _app.register_blueprint(api)
    _app.run(host=host, port=port)


if __name__ == "__main__":
    main()
