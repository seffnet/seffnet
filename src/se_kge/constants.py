# -*- coding: utf-8 -*-

"""Constants for ``se_kge``."""

import os

__all__ = [
    'DRUGBANK_NAMESPACE',
    'PUBCHEM_NAMESPACE',
    'UNIPROT_NAMESPACE',
    'HERE',
    'RESOURCES',
    'DEFAULT_DRUGBANK_PICKLE',
    'DEFAULT_SIDER_PICKLE',
    'DEFAULT_EMBEDDINGS_PATH',
    'DEFAULT_GRAPH_PATH',
    'DEFAULT_MAPPING_PATH',
    'DEFAULT_MODEL_PATH',
]

DRUGBANK_NAMESPACE = 'drugbank'
PUBCHEM_NAMESPACE = 'pubchem.compound'
UNIPROT_NAMESPACE = 'uniprot'

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'resources'))

DEFAULT_DRUGBANK_PICKLE = os.path.join(RESOURCES, 'basic_graphs', 'drugbank_graph.pickle')
DEFAULT_SIDER_PICKLE = os.path.join(RESOURCES, 'basic_graphs', 'sider_graph.pickle')

DEFAULT_EMBEDDINGS_PATH = os.path.abspath(os.path.join(
    RESOURCES, "predictive_model", "070819_node2vec_embeddings_complete01.embeddings",
))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(
    RESOURCES, "predictive_model", "070819_node2vec_model_complete01.pkl",
))
DEFAULT_GRAPH_PATH = os.path.abspath(os.path.join(
    RESOURCES, "chemsim_50_graphs", "fullgraph_with_chemsim_50.edgelist",
))
DEFAULT_MAPPING_PATH = os.path.abspath(os.path.join(
    RESOURCES, "mapping", "fullgraph_nodes_mapping.tsv",
))
