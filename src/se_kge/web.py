# -*- coding: utf-8 -*-

"""A RESTful API for ``se_kge``."""

import logging
import os

import networkx as nx
import pandas as pd
from bionev.utils import load_embedding
from flasgger import Swagger
from flask import Blueprint, Flask, current_app, jsonify, request, url_for
from sklearn.externals import joblib

from se_kge.constants import RESOURCES
from se_kge.find_relations import find_new_relations

__all__ = [
    'get_app',
    'api',
]

logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)


@api.route('/')
def home():
    test_url = url_for('.find', entity='85')
    return f"""
    <html>
    <body>
    Check the <a href="{test_url}">results</a> for 3-Hydroxy-4-trimethylammoniobutanoate
    (pubchem:85)
    </body>
    </html>
    """


@api.route('/find/<entity_identifier>')
def find(entity_identifier):
    """Find new entities.

    ---
    parameters:
      - name: entity
        in: path
        description: The entity's CURIE
        required: true
        type: string
      - name: entity_type
        in: query
        description: The type of the entities for the incedent relations that get predicted
        required: false
        type: string
      - name: k
        in: query
        description: The number of predictions to return
        required: false
        type: integer

    """
    entity_type = request.args.get('entity_type', 'phenotype')
    k = request.args.get('k', 30, type=int)

    res = find_new_relations(
        entity_identifier=entity_identifier,
        embeddings=current_app.config['embeddings'],
        node_mapping=current_app.config['node_mapping'],
        saved_model=current_app.config['model'],
        graph=current_app.config['graph'],
        entity_type=entity_type,
        k=k,
    )

    return jsonify(res)


def get_app() -> Flask:
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
    app.config.update({
        'embeddings': embeddings_node2vec,
        'model': model,
        'graph': graph,
        'node_mapping': node_mapping,
    })

    Swagger(app)

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    _app = get_app()
    _app.register_blueprint(api)
    _app.run()
