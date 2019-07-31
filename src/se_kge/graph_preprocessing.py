# -*- coding: utf-8 -*-

import bio2bel_drugbank
import bio2bel_sider
import pybel.dsl
import pybel

from tqdm import tqdm
import pandas as pd
import networkx as nx


def get_sider_graph():
    sider_manager = bio2bel_sider.Manager()
    if sider_manager.is_populated() == False:
        sider_manager.populate()
    sider_graph = sider_manager.to_bel()
    return sider_graph

def get_drugbank_graph():
    drugbank_manager = bio2bel_drugbank.Manager()
    if drugbank_manager.is_populated() == False:
        drugbank_manager.populate()
    drugbank_graph = drugbank_manager.to_bel()
    drugbank_manager.enrich_targets(drugbank_graph)
    return drugbank_graph

def combine_pubchem_drugbank(pubchem_drugbank_mapping_file, drugbank_graph, sider_graph):
    drugbank_pubchem_mapping = pd.read_csv(
        pubchem_drugbank_mapping_file, sep=",",
        index_col=False, dtype={'PubchemID': str, 'Smiles': str, 'DrugbankID': str})
    drugbank_to_pubchem = {}
    for ind, row in tqdm(drugbank_pubchem_mapping.iterrows(), desc='create pubchem-drugbank mapping dictionary'):
        drugbank_to_pubchem[pybel.dsl.Abundance(namespace='drugbank', name=row['DrugbankName'])] = pybel.dsl.Abundance(namespace='pubchem', identifier=row['PubchemID'])
    drugbank_relabel = nx.relabel_nodes(drugbank_graph, drugbank_to_pubchem)
    rm_nodes = []
    for node in drugbank_relabel.nodes():
        if node.namespace == 'drugbank':
            rm_nodes.append(node)
    print('Number of nodes that were not relabeled %d' % len(rm_nodes))
    for node in tqdm(rm_nodes, desc='Removing nodes that were not relabeled'):
        drugbank_relabel.remove_node(node)
    full_graph = sider_graph + drugbank_relabel
    print('The number of nodes in the combined graph is %d' % len(full_graph.nodes()))
    return full_graph
