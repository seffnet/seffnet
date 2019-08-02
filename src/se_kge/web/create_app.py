# -*- coding: utf-8 -*-

"""Create the flask application for ``se_kge``."""

import logging
import os

import networkx as nx
import pandas as pd
from bionev.utils import load_embedding
from flasgger import Swagger
from flask import Flask
from flask_bootstrap import Bootstrap
from sklearn.externals import joblib

from ..constants import RESOURCES

__all__ = [
    'create_app',
]

logger = logging.getLogger('se_kge.web')


def create_app() -> Flask:
    """Make the SE_KGE web app."""
    # Load embedding
    embedding_filepath = os.path.join(RESOURCES, "240719_node2vec_fullgraph.embeddings")
    logger.info(f'Loading embeddings from {embedding_filepath}')
    assert os.path.exists(embedding_filepath)
    embeddings_node2vec = load_embedding(embedding_filepath)

    # Load model
    model_filepath = os.path.join(RESOURCES, "prediction_model_node2vec_final.pkl")
    logger.info(f'Loading model from {model_filepath}')
    assert os.path.exists(model_filepath)
    model = joblib.load(model_filepath)

    # Load graph
    graph_filepath = os.path.join(RESOURCES, "fullgraph.edgelist")
    logger.info(f'Loading graph from {graph_filepath}')
    assert os.path.exists(graph_filepath)
    graph = nx.read_edgelist(graph_filepath)

    # Load node mapping
    node_mapping_filepath = os.path.join(RESOURCES, "fullgraph_nodes_mapping.tsv")
    logger.info(f'Loading node mapping from {node_mapping_filepath}')
    assert os.path.exists(node_mapping_filepath)
    node_mapping = pd.read_csv(node_mapping_filepath, sep=',')

    app = Flask(__name__)
    app.secret_key = os.urandom(8)
    app.config.update({
        'embeddings': embeddings_node2vec,
        'model': model,
        'graph': graph,
        'node_mapping': node_mapping,
    })

    Swagger(app)
    Bootstrap(app)

    return app
