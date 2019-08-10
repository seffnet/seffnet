# -*- coding: utf-8 -*-

"""Command line interface for ``se_kge``."""

import datetime
import getpass
import json
import logging
import random
import sys

import click
import networkx as nx
from bionev import pipeline
from bionev.embed_train import embedding_training

from .constants import DEFAULT_GRAPH_PATH
from .find_relations import RESULTS_TYPE_TO_NAMESPACE
from .optimization import (
    deepwalk_optimization, grarep_optimization, hope_optimization, line_optimization,
    node2vec_optimization, sdne_optimization,
)
from .utils import study_to_json


@click.group()
def main():
    """Side Effects Knowledge Graph Embeddings."""


@main.command()
@click.option('--input-path', default=DEFAULT_GRAPH_PATH,
              help='Input graph file. Only accepted edgelist format.')
@click.option('--training', help='training graph file. Only accepted edgelist format.')
@click.option('--testing', help='testing graph file. Only accepted edgelist format.')
@click.option('--method', required=True,
              type=click.Choice(['node2vec', 'DeepWalk', 'HOPE', 'GraRep', 'LINE', 'SDNE']),
              help='The NRL method to train the model')
@click.option('--trials', default=50, type=int, help='the number of trials done to optimize hyperparameters')
@click.option('--dimensions-range', default=(100, 300), type=(int, int), help='the range of dimensions to be optimized')
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
        dimensions_range,
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


@main.command()
@click.option('--input-path', default=DEFAULT_GRAPH_PATH, help='Input graph file. Only accepted edgelist format.')
@click.option('--training', help='training graph file. Only accepted edgelist format.')
@click.option('--testing', help='testing graph file. Only accepted edgelist format.')
@click.option('--evaluation', is_flag=True, help='If true, a testing set will be used to evaluate model.')
@click.option('--evaluation-file', type=click.File('w'), default=sys.stdout,
              help='The path to save evaluation results.')
@click.option('--embeddings-path', help='The path to save the embeddings file')
@click.option('--model-path',
              help='The path to save the prediction model')
@click.option('--seed', type=int, default=random.randint(1, 10000000))
@click.option('--method', required=True,
              type=click.Choice(['node2vec', 'DeepWalk', 'HOPE', 'GraRep', 'LINE', 'SDNE']),
              help='The NRL method to train the model')
@click.option('--dimensions', type=int, default=200,
              help='The dimensions of embeddings.')
@click.option('--number-walks', type=int, default=8,
              help='The number of walks for random-walk methods.')
@click.option('--walk-length', type=int, default=32,
              help='The walk length for random-walk methods.')
@click.option('--window-size', type=int, default=3,
              help='The window size for random-walk methods.')
@click.option('--p', type=float, default=1.5,
              help='The p parameter for node2vec.')
@click.option('--q', type=float, default=0.8,
              help='The q parameter for node2vec.')
@click.option('--alpha', type=float, default=0.3,
              help='The alpha parameter for SDNE')
@click.option('--beta', type=int, default=2,
              help='The beta parameter for SDNE')
@click.option('--epochs', type=float, default=30,
              help='The epochs for deep learning methods')
@click.option('--kstep', type=float, default=30,
              help='The kstep parameter for GraRep')
@click.option('--order', default=2, type=click.Choice([1, 2, 3]),
              help='The order parameter for LINE')
def train(
        input_path,
        training,
        testing,
        evaluation,
        evaluation_file,
        embeddings_path,
        model_path,
        seed,
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
):
    """Train my model."""
    if evaluation:
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

    else:
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
        click.echo('Training is finished.')


@main.command()
@click.argument('curie')
@click.option('-n', '--number-predictions', type=int, default=30)
@click.option('-t', '--result-type', type=click.Choice(RESULTS_TYPE_TO_NAMESPACE))
def predict(curie: str, number_predictions: int, result_type: str):
    """Predict for a given entity."""
    from .default_predictor import predictor

    if number_predictions <= 0:
        number_predictions = None

    results = predictor.find_new_relations(
        node_curie=curie,
        k=number_predictions,
        results_type=result_type,
    )
    for result in results['predictions']:
        click.echo(json.dumps(result, indent=2, sort_keys=True))


@main.command()
@click.option('--host')
@click.option('--port', type=int)
def web(host, port):
    """Run the RESTful API."""
    from .web import create_app, api
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('se_kge.web').setLevel(logging.INFO)
    _app = create_app()
    _app.register_blueprint(api)
    _app.run(host=host, port=port)


@main.command()
def rebuild():
    """Build all resources from scratch."""
    from pybel.struct import count_functions
    from .graph_preprocessing import get_drugbank_graph, get_sider_graph

    click.secho('Rebuilding DrugBank', fg='blue', bold=True)
    drugbank_graph = get_drugbank_graph(rebuild=True, drug_namespace='pubchem.compound')
    click.echo(drugbank_graph.summary_str())
    click.echo(str(count_functions(drugbank_graph)))

    click.secho('Rebuilding SIDER', fg='blue', bold=True)
    sider_graph = get_sider_graph(rebuild=True)
    click.echo(sider_graph.summary_str())
    click.echo(str(count_functions(sider_graph)))


if __name__ == "__main__":
    main()
