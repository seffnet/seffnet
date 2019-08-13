# -*- coding: utf-8 -*-

"""Command line interface for ``se_kge``."""

import json
import logging
import random
import sys

import click
import networkx as nx

from .constants import DEFAULT_GRAPH_PATH
from .find_relations import RESULTS_TYPE_TO_NAMESPACE
from .graph_preprocessing import get_mapped_graph
from .utils import do_evaluation, do_optimization, split_training_testing_sets, train_model

INPUT_PATH = click.option('--input-path', default=DEFAULT_GRAPH_PATH,
                          help='Input graph file. Only accepted edgelist format.')
TRAINING_PATH = click.option('--training-path', help='training graph file. Only accepted edgelist format.')
TESTING_PATH = click.option('--testing-path', help='testing graph file. Only accepted edgelist format.')
METHOD = click.option('--method', required=True,
                      type=click.Choice(['node2vec', 'DeepWalk', 'HOPE', 'GraRep', 'LINE', 'SDNE']),
                      help='The NRL method to train the model')
SEED = click.option('--seed', type=int, default=random.randint(1, 10000000))


@click.group()
def main():
    """Side Effects Knowledge Graph Embeddings."""


@main.command()
@INPUT_PATH
@TRAINING_PATH
@TESTING_PATH
@METHOD
@SEED
@click.option('--trials', default=50, type=int, help='the number of trials done to optimize hyperparameters')
@click.option('--dimensions-range', default=(100, 300), type=(int, int), help='the range of dimensions to be optimized')
@click.option('--storage', help="SQL connection string for study database. Example: sqlite:///optuna.db")
@click.option('--name', help="Name for the study")
@click.option('-o', '--output', type=click.File('w'), help="Output study summary", default=sys.stdout)
def optimize(
        input_path,
        training_path,
        testing_path,
        method,
        trials,
        dimensions_range,
        seed,
        storage,
        name,
        output,
):
    """Run the optimization pipeline for a given method and graph."""
    do_optimization(
        input_path=input_path,
        training_path=training_path,
        testing_path=testing_path,
        method=method,
        trials=trials,
        storage=storage,
        dimensions_range=dimensions_range,
        name=name,
        output=output,
        seed=seed,
    )


@main.command()
@INPUT_PATH
@TRAINING_PATH
@TESTING_PATH
@SEED
@METHOD
@click.option('--evaluation', is_flag=True, help='If true, a testing set will be used to evaluate model.')
@click.option('--evaluation-file', type=click.File('w'), default=sys.stdout,
              help='The path to save evaluation results.')
@click.option('--embeddings-path', help='The path to save the embeddings file')
@click.option('--model-path',
              help='The path to save the prediction model')
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
        training_path,
        testing_path,
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
        results = do_evaluation(
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
            seed=seed,
            embeddings_path=embeddings_path,
            model_path=model_path,
            evaluation_file=evaluation_file
        )
        click.echo('Training is finished.')
        click.echo(results)
    else:
        train_model(
            input_path=input_path,
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
            model_path=model_path,
            embeddings_path=embeddings_path,
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
    from pybel.struct import count_functions, count_namespaces
    from .graph_preprocessing import get_drugbank_graph, get_sider_graph, get_combined_sider_drugbank
    try:
        from se_kge.chemical_similarities import get_similarity_graph, cluster_chemicals
    except:
        raise Exception('You need rdkit package to rebuild the graphs')

    def _echo_graph(graph):
        click.echo(graph.summary_str())
        click.echo(str(count_functions(graph)))
        click.echo(str(count_namespaces(graph)))
        click.echo(str(graph.number_of_edges()))

    click.secho('Rebuilding DrugBank', fg='blue', bold=True)
    drugbank_graph = get_drugbank_graph(rebuild=True, drug_namespace='pubchem.compound')
    click.echo(drugbank_graph.summary_str())
    _echo_graph(drugbank_graph)

    click.secho('Rebuilding SIDER', fg='blue', bold=True)
    sider_graph = get_sider_graph(rebuild=True)
    _echo_graph(sider_graph)

    click.secho('Rebuilding DrugBank-SIDER combined graph', fg='blue', bold=True)
    fullgraph = get_combined_sider_drugbank(
        rebuild=True,
        sider_graph_path=sider_graph,
        drugbank_graph_path=drugbank_graph,
    )
    _echo_graph(fullgraph)

    click.secho('Rebuilding combined graph with node_ids', fg='blue', bold=True)
    get_mapped_graph(graph_path=fullgraph, rebuild=True)
    click.echo('Mapped graph and mapping dataframe are created!')

    click.secho('Rebuilding combined graph with chemical similarities', fg='blue', bold=True)
    fullgraph_with_chemsim = get_similarity_graph(rebuild=True)
    _echo_graph(fullgraph_with_chemsim)

    click.secho('Reclustering chemicals', fg='blue', bold=True)
    cluster_chemicals(rebuild=True)
    click.echo('Clustered chemicals dataframe is created!')

    click.secho('Rebuilding training and testing sets', fg='blue', bold=True)
    g_train, g_test = split_training_testing_sets(rebuild=True)
    click.echo(nx.info(g_train))
    click.echo(nx.info(g_test))


if __name__ == "__main__":
    main()
