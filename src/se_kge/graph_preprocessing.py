# -*- coding: utf-8 -*-

"""Pre-processing of Graphs used for NRL models."""

import bio2bel_drugbank
import bio2bel_sider
import networkx as nx
import pandas as pd
import pybel
import pybel.dsl
from tqdm import tqdm


def get_sider_graph():
    """
    Get the SIDER graph.

    :return: BELGraph
    """
    sider_manager = bio2bel_sider.Manager()
    if sider_manager.is_populated() is False:
        sider_manager.populate()
    sider_graph = sider_manager.to_bel()
    return sider_graph


def get_drugbank_graph():
    """
    Get the DrugBank graph.

    :return: BELGraph
    """
    drugbank_manager = bio2bel_drugbank.Manager()
    if drugbank_manager.is_populated() is False:
        drugbank_manager.populate()
    drugbank_graph = drugbank_manager.to_bel()
    return drugbank_graph


def combine_pubchem_drugbank(pubchem_drugbank_mapping_file, drugbank_graph, sider_graph):
    """
    Combine the SIDER and DrugBank graphs.

    :param pubchem_drugbank_mapping_file: a tsv file with mappings between pubchemid and drugbankid
    :param drugbank_graph: the drugbank graph
    :param sider_graph: the sider graph
    :return: BELGraph
    """
    drugbank_pubchem_mapping = pd.read_csv(
        pubchem_drugbank_mapping_file, sep="\t",
        index_col=False, dtype={'PubchemID': str, 'Smiles': str, 'DrugbankID': str})
    drugbank_pubchem_mapping = drugbank_pubchem_mapping.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    drugbank_to_pubchem = {}
    for ind, row in tqdm(drugbank_pubchem_mapping.iterrows(), desc='create pubchem-drugbank mapping dictionary'):
        drugbank_to_pubchem[pybel.dsl.Abundance(
            namespace='drugbank',
            name=row['DrugbankName'],
            identifier=row['DrugbankID'])] = pybel.dsl.Abundance(
            namespace='pubchem',
            identifier=row['PubchemID'])
    drugbank_relabel = nx.relabel_nodes(drugbank_graph, drugbank_to_pubchem)
    rm_nodes = []
    for node in drugbank_relabel.nodes():
        if node.namespace == 'drugbank':
            rm_nodes.append(node)
    for node in tqdm(rm_nodes, desc='Removing nodes that were not relabeled'):
        drugbank_relabel.remove_node(node)
    full_graph = sider_graph + drugbank_relabel
    full_graph.remove_nodes_from(list(nx.isolates(full_graph)))
    return full_graph
