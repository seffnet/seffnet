# -*- coding: utf-8 -*-

"""Command line interface for :mod:`seffnet`."""

import json
import logging
import random
import sys

import click
import joblib
import networkx as nx
import numpy as np

import bionev.OpenNE.graph as og
from bionev.pipeline import create_prediction_model
from .constants import DEFAULT_FULLGRAPH_PICKLE, DEFAULT_GRAPH_PATH
from .find_relations import RESULTS_TYPE_TO_NAMESPACE
from .graph_preprocessing import get_mapped_graph
from .utils import do_evaluation, do_optimization, repeat_experiment, split_training_testing_sets, train_model

INPUT_PATH = click.option('--input-path', default=DEFAULT_GRAPH_PATH,
                          help='Input graph file. Only accepted edgelist format.')
TRAINING_PATH = click.option('--training-path', help='training graph file. Only accepted edgelist format.')
TESTING_PATH = click.option('--testing-path', help='testing graph file. Only accepted edgelist format.')
METHOD = click.option('--method', required=True,
                      type=click.Choice(['node2vec', 'DeepWalk', 'HOPE', 'GraRep', 'LINE', 'SDNE']),
                      help='The NRL method to train the model')
SEED = click.option('--seed', type=int, default=random.randrange(sys.maxsize))
DIMENSIONS = click.option('--dimensions', type=int, default=200, help='The dimensions of embeddings.')
NUMBER_WALKS = click.option('--number-walks', type=int, default=8, help='The number of walks for random-walk methods.')
WALK_LENGTH = click.option('--walk-length', type=int, default=32, help='The walk length for random-walk methods.')
WINDOW_SIZE = click.option('--window-size', type=int, default=3, help='The window size for random-walk methods.')
P = click.option('--p', type=float, default=1.5, help='The p parameter for node2vec.')
Q = click.option('--q', type=float, default=0.8, help='The q parameter for node2vec.')
ALPHA = click.option('--alpha', type=float, default=0.3, help='The alpha parameter for SDNE')
BETA = click.option('--beta', type=int, default=2, help='The beta parameter for SDNE')
EPOCHS = click.option('--epochs', type=int, default=30, help='The epochs for deep learning methods')
KSTEP = click.option('--kstep', type=int, default=30, help='The kstep parameter for GraRep')
ORDER = click.option('--order', default=2, type=int, help='The order parameter for LINE. Could be 1, 2 or 3')
EVALUATION_FILE = click.option('--evaluation-file', type=click.File('w'), default=sys.stdout,
                               help='The path to save evaluation results.')
PREDICTION_TASK = click.option('--prediction-task', default='link_prediction',
                               type=click.Choice(['link_prediction', 'node_classification']),
                               required=True,
                               help='The prediction task for the model')
LABELS_FILE = click.option('--labels-file', default='', help='The labels file for node classification')
TRAINING_MODEL_PATH = click.option('--training-model-path', help='The path to save the model used for training')
PREDICTIVE_MODEL_PATH = click.option('--predictive-model-path', help='The path to save the prediction model')
EMBEDDINGS_PATH = click.option('--embeddings-path', help='The path to save the embeddings file')
CLASSIFIER_TYPE = click.option('--classifier-type', type=click.Choice(['LR', 'EN', 'SVM', 'RF']),
                               help='Choose type of classifier for predictive model')
WEIGHTED = click.option('--weighted', is_flag=False, help='True if graph is weighted.')


@click.group()
def main():
    """Side Effects Knowledge Graph Embeddings."""


@main.command()
@INPUT_PATH
@TRAINING_PATH
@TESTING_PATH
@METHOD
@SEED
@PREDICTION_TASK
@LABELS_FILE
@click.option('--trials', default=50, type=int, help='the number of trials done to optimize hyperparameters')
@click.option('--dimensions-range', default=(100, 300), type=(int, int), help='the range of dimensions to be optimized')
@click.option('--storage', help="SQL connection string for study database. Example: sqlite:///optuna.db")
@click.option('--name', help="Name for the study")
@click.option('-o', '--output', type=click.File('w'), help="Output study summary", default=sys.stdout)
@WEIGHTED
@CLASSIFIER_TYPE
def optimize(
    input_path,
    training_path,
    testing_path,
    seed,
    prediction_task,
    labels_file,
    method,
    trials,
    dimensions_range,
    storage,
    name,
    output,
    classifier_type,
    weighted,
):
    """Run the optimization pipeline for a given method and graph."""
    do_optimization(
        input_path=input_path,
        training_path=training_path,
        testing_path=testing_path,
        prediction_task=prediction_task,
        labels_file=labels_file,
        method=method,
        trials=trials,
        storage=storage,
        dimensions_range=dimensions_range,
        name=name,
        output=output,
        classifier_type=classifier_type,
        weighted=weighted,
        study_seed=seed,
    )


@main.command()
@INPUT_PATH
@TRAINING_PATH
@TESTING_PATH
@SEED
@METHOD
@click.option('--evaluation', is_flag=True, help='If true, a testing set will be used to evaluate model.')
@EVALUATION_FILE
@EMBEDDINGS_PATH
@PREDICTIVE_MODEL_PATH
@TRAINING_MODEL_PATH
@DIMENSIONS
@NUMBER_WALKS
@WALK_LENGTH
@WINDOW_SIZE
@P
@Q
@ALPHA
@BETA
@EPOCHS
@KSTEP
@ORDER
@CLASSIFIER_TYPE
@WEIGHTED
@PREDICTION_TASK
@LABELS_FILE
def train(
    input_path,
    training_path,
    testing_path,
    evaluation,
    evaluation_file,
    embeddings_path,
    predictive_model_path,
    training_model_path,
    classifier_type,
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
    weighted,
    prediction_task,
    labels_file,
):
    """Train my model."""
    np.random.seed(seed)
    random.seed(seed)

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
            embeddings_path=embeddings_path,
            predictive_model_path=predictive_model_path,
            training_model_path=training_model_path,
            evaluation_file=evaluation_file,
            classifier_type=classifier_type,
            weighted=weighted,
            prediction_task=prediction_task,
            labels_file=labels_file,
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
            predictive_model_path=predictive_model_path,
            training_model_path=training_model_path,
            embeddings_path=embeddings_path,
            weighted=weighted,
            labels_file=labels_file,
            prediction_task=prediction_task,
        )
        click.echo('Training is finished.')


@main.command()
@click.option('--updated-graph', default=DEFAULT_GRAPH_PATH, help='an edgelist containing the graph with new nodes')
@click.option('--chemicals-list', help='a file containing list of chemicals to update the model with')
@click.option('--old-graph', default=DEFAULT_FULLGRAPH_PICKLE, help='The graph needed  to be updated. In pickle format')
@click.option('--updated-graph-path', required=True, help='The path to save the updated fullgraph')
@click.option('--chemsim-graph-path', required=True, help='The path to save the chemical similarity graph')
@click.option('--training-model-path', required=True, help='The path to save the model used for training')
@click.option('--new-training-model-path', required=True,
              help='the path of the updated training model')
@EMBEDDINGS_PATH
@PREDICTIVE_MODEL_PATH
@SEED
def update(
    updated_graph,
    old_graph,
    chemicals_list,
    updated_graph_path,
    chemsim_graph_path,
    training_model_path,
    new_training_model_path,
    embeddings_path,
    predictive_model_path,
    seed,
):
    """Update node2vec training model."""
    np.random.seed(seed)
    random.seed(seed)

    if chemicals_list is not None:
        new_chemicals = [line.rstrip('\n') for line in open(chemicals_list)]
        try:
            from seffnet.chemical_similarities import add_new_chemicals
        except Exception:
            raise Exception('You need RDKit to update model')
        click.secho('Updating graph', fg='blue', bold=True)
        pickled_graph_path = updated_graph_path.split('.')[0] + '.pickle'
        new_graph = add_new_chemicals(chemicals_list=new_chemicals, graph=old_graph,
                                      updated_graph_path=updated_graph_path, chemsim_graph_path=chemsim_graph_path,
                                      pickled_graph_path=pickled_graph_path)
        graph = og.Graph()
        graph.read_g(new_graph)
    else:
        click.secho('Loading graph', fg='blue', bold=True)
        graph = og.Graph()
        graph.read_edgelist(updated_graph, weighted=False)
    click.secho('Loading training model', fg='blue', bold=True)
    model = joblib.load(training_model_path)
    click.secho('Updating training model', fg='blue', bold=True)
    model.update_model(graph)
    joblib.dump(model, new_training_model_path)
    if embeddings_path is not None:
        model.save_embeddings(embeddings_path)
    if predictive_model_path is not None:
        click.secho('Building predictive model', fg='blue', bold=True)
        original_graph = graph.G
        create_prediction_model(
            embeddings=model.get_embeddings(),
            original_graph=original_graph,
            save_model=predictive_model_path
        )
    click.secho('Process is complete', fg='blue', bold=True)


@main.command()
@INPUT_PATH
@TRAINING_PATH
@TESTING_PATH
@METHOD
@EVALUATION_FILE
@DIMENSIONS
@NUMBER_WALKS
@WALK_LENGTH
@WINDOW_SIZE
@P
@Q
@ALPHA
@BETA
@EPOCHS
@KSTEP
@ORDER
@click.option('--n', default=10, help='number of repeats.')
@SEED
def repeat(
    input_path,
    training_path,
    testing_path,
    evaluation_file,
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
    n,
    seed,
):
    """Repeat training n times."""
    np.random.seed(seed)
    random.seed(seed)

    results = repeat_experiment(
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
        n=n,
        evaluation_file=evaluation_file,
    )
    click.echo(results)


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
    logging.getLogger('seffnet.web').setLevel(logging.INFO)
    _app = create_app()
    _app.register_blueprint(api)
    _app.run(host=host, port=port)


@main.command()
def rebuild():
    """Build all resources from scratch."""
    from pybel.struct import count_functions, count_namespaces
    from .graph_preprocessing import get_drugbank_graph, get_sider_graph, get_combined_sider_drugbank
    try:
        from .chemical_similarities import get_similarity_graph, cluster_chemicals, get_combined_graph_similarity
    except Exception:
        raise Exception('You need RDKit to rebuild the graphs')

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
    chemsim_graph = get_similarity_graph(rebuild=True)
    fullgraph_with_chemsim = get_combined_graph_similarity(fullgraph, chemsim_graph)
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
