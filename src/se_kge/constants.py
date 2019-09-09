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
    'DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE',
    'DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST',
    'DEFAULT_EMBEDDINGS_PATH',
    'DEFAULT_GRAPH_PATH',
    'DEFAULT_MAPPING_PATH',
    'DEFAULT_MODEL_PATH',
    'DEFAULT_CHEMICALS_MAPPING_PATH',
    'DEFAULT_FULLGRAPH_PICKLE',
    'DEFAULT_CHEMSIM_PICKLE',
    'DEFAULT_CLUSTERED_CHEMICALS',
    'DEFAULT_TRAINING_SET',
    'DEFAULT_TESTING_SET',
]

DRUGBANK_NAMESPACE = 'drugbank'
PUBCHEM_NAMESPACE = 'pubchem.compound'
UNIPROT_NAMESPACE = 'uniprot'

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'resources'))

BASIC_GRAPHS = os.path.join(RESOURCES, 'basic_graphs')
DEFAULT_DRUGBANK_PICKLE = os.path.join(BASIC_GRAPHS, 'drugbank_graph.pickle')
DEFAULT_SIDER_PICKLE = os.path.join(BASIC_GRAPHS, 'sider_graph.pickle')
DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE = os.path.join(BASIC_GRAPHS, 'fullgraph_without_chemsim.pickle')
DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST = os.path.join(BASIC_GRAPHS, 'fullgraph_without_chemsim.edgelist')

PREDICTIVE_MODEL = os.path.join(RESOURCES, "predictive_model")
DEFAULT_EMBEDDINGS_PATH = os.path.join(PREDICTIVE_MODEL, "0809_line_emb.embeddings")
DEFAULT_MODEL_PATH = os.path.join(PREDICTIVE_MODEL, "0809_line_model.pkl")

MAPPING = os.path.join(RESOURCES, "mapping")
DEFAULT_CHEMICALS_MAPPING_PATH = os.path.join(MAPPING, "drugbank_pubchem_mapping.tsv")
DEFAULT_MAPPING_PATH = os.path.join(MAPPING, "fullgraph_nodes_mapping.tsv")
DEFAULT_CLUSTERED_CHEMICALS = os.path.join(MAPPING, "clustered_chemicals.tsv")

CHEMSIM_50_GRAPHS = os.path.join(RESOURCES, "chemsim_50_graphs")
DEFAULT_CHEMSIM_PICKLE = os.path.join(CHEMSIM_50_GRAPHS, "chemsim_graph_50.pickle")
DEFAULT_FULLGRAPH_PICKLE = os.path.join(CHEMSIM_50_GRAPHS, "fullgraph_with_chemsim_50.pickle")
DEFAULT_GRAPH_PATH = os.path.join(BASIC_GRAPHS, "fullgraph_with_chemsim.edgelist")
DEFAULT_TRAINING_SET = os.path.join(CHEMSIM_50_GRAPHS, "training_edgelist_50.edgelist")
DEFAULT_TESTING_SET = os.path.join(CHEMSIM_50_GRAPHS, "testing_edgelist_50.edgelist")
