# -*- coding: utf-8 -*-

"""Constants for :mod:`seffnet`."""

import os

__all__ = [
    'DRUGBANK_NAMESPACE',
    'PUBCHEM_NAMESPACE',
    'UNIPROT_NAMESPACE',
    'HERE',
    'RESOURCES',
    'DEFAULT_DRUGBANK_PICKLE',
    'DEFAULT_SIDER_PICKLE',
    'DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE',
    'DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST',
    'DEFAULT_EMBEDDINGS_PATH',
    'DEFAULT_GRAPH_PATH',
    'DEFAULT_MAPPING_PATH',
    'DEFAULT_PREDICTIVE_MODEL_PATH',
    'DEFAULT_CHEMICALS_MAPPING_PATH',
    'DEFAULT_FULLGRAPH_PICKLE',
    'DEFAULT_CHEMSIM_PICKLE',
    'DEFAULT_CLUSTERED_CHEMICALS',
    'DEFAULT_TRAINING_SET',
    'DEFAULT_TESTING_SET',
    'DEFAULT_POTENCY_MAPPING_PATH',
    'DEFAULT_DRUGBANK_WEIGHTED_PICKLE',
    'DEFAULT_CHEMSIM_WEIGHTED_PICKLE',
    'DEFAULT_SIDER_WEIGHTED_PICKLE',
    'DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE',
    'DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_EDGELIST',
    'DEFAULT_GRAPH_WEIGHTED_PATH',
    'DEFAULT_WEIGHTED_FULLGRAPH_PICKLE',
    'DEFAULT_WEIGHTED_TESTING_SET',
    'DEFAULT_WEIGHTED_TRAINING_SET'
]

DRUGBANK_NAMESPACE = 'drugbank'
PUBCHEM_NAMESPACE = 'pubchem.compound'
UNIPROT_NAMESPACE = 'uniprot'

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'resources'))

BASIC_GRAPHS = os.path.join(RESOURCES, 'basic_graphs')
DEFAULT_DRUGBANK_PICKLE = os.path.join(BASIC_GRAPHS, 'drugbank_graph.pickle')
DEFAULT_DRUGBANK_WEIGHTED_PICKLE = os.path.join(BASIC_GRAPHS, 'drugbank_weighted_graph.pickle')
DEFAULT_SIDER_PICKLE = os.path.join(BASIC_GRAPHS, 'sider_graph.pickle')
DEFAULT_SIDER_WEIGHTED_PICKLE = os.path.join(BASIC_GRAPHS, 'sider_weighted_graph.pickle')
DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE = os.path.join(BASIC_GRAPHS, 'fullgraph_without_chemsim.pickle')
DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST = os.path.join(BASIC_GRAPHS, 'fullgraph_without_chemsim.edgelist')
DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE = os.path.join(
    BASIC_GRAPHS,
    'fullgraph_weighted_without_chemsim.pickle'
)
DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_EDGELIST = os.path.join(
    BASIC_GRAPHS,
    'fullgraph_weighted_without_chemsim.edgelist'
)

PREDICTIVE_MODELS = os.path.join(RESOURCES, "predictive_models")
EMBEDDINGS = os.path.join(RESOURCES, "embeddings")
TRAINING_MODELS = os.path.join(RESOURCES, "training_models")
DEFAULT_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS, "0411_weighted_node2vec_emb.embeddings")
DEFAULT_PREDICTIVE_MODEL_PATH = os.path.join(PREDICTIVE_MODELS, "0411_weighted_node2vec_predictive_model.pkl")

MAPPING = os.path.join(RESOURCES, "mapping")
DEFAULT_CHEMICALS_MAPPING_PATH = os.path.join(MAPPING, "chemicals_mapping.tsv")
DEFAULT_MAPPING_PATH = os.path.join(MAPPING, "fullgraph_nodes_mapping.tsv")
DEFAULT_CLUSTERED_CHEMICALS = os.path.join(MAPPING, "clustered_chemicals.tsv")
DEFAULT_POTENCY_MAPPING_PATH = os.path.join(MAPPING, "potency_mapping.tsv")

CHEMSIM_50_GRAPHS = os.path.join(RESOURCES, "chemsim_50_graphs")
DEFAULT_CHEMSIM_PICKLE = os.path.join(BASIC_GRAPHS, "chemsim_graph.pickle")
DEFAULT_CHEMSIM_WEIGHTED_PICKLE = os.path.join(BASIC_GRAPHS, "chemsim_weighted_graph.pickle")
DEFAULT_FULLGRAPH_PICKLE = os.path.join(BASIC_GRAPHS, "fullgraph_with_chemsim.pickle")
DEFAULT_WEIGHTED_FULLGRAPH_PICKLE = os.path.join(BASIC_GRAPHS, "fullgraph_weighted_with_chemsim.pickle")
DEFAULT_GRAPH_PATH = os.path.join(BASIC_GRAPHS, "fullgraph_with_chemsim.edgelist")
DEFAULT_GRAPH_WEIGHTED_PATH = os.path.join(BASIC_GRAPHS, "fullgraph_weighted_with_chemsim.edgelist")
DEFAULT_TRAINING_SET = os.path.join(BASIC_GRAPHS, "training_set.edgelist")
DEFAULT_TESTING_SET = os.path.join(BASIC_GRAPHS, "testing_set.edgelist")
DEFAULT_WEIGHTED_TRAINING_SET = os.path.join(BASIC_GRAPHS, "weighted_training_set.edgelist")
DEFAULT_WEIGHTED_TESTING_SET = os.path.join(BASIC_GRAPHS, "weighted_testing_set.edgelist")