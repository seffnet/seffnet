# -*- coding: utf-8 -*-

"""Pre-processing of Graphs used for NRL models."""

import os

import networkx as nx
import pandas as pd
import pybel
import pybel.dsl
from defusedxml import ElementTree
from tqdm import tqdm

import bio2bel_drugbank
import bio2bel_sider
from .constants import (
    DEFAULT_DRUGBANK_PICKLE, DEFAULT_MAPPING_PATH, DEFAULT_SIDER_PICKLE, DRUGBANK_NAMESPACE, PUBCHEM_NAMESPACE,
    RESOURCES, UNIPROT_NAMESPACE,
)
from .get_url_requests import cid_to_synonyms, get_gene_names, smiles_to_cid


def get_sider_graph() -> pybel.BELGraph:
    """Get the SIDER graph."""
    if os.path.exists(DEFAULT_SIDER_PICKLE):
        return pybel.from_pickle(DEFAULT_SIDER_PICKLE)

    sider_manager = bio2bel_sider.Manager()
    if not sider_manager.is_populated():
        sider_manager.populate()
    sider_graph = sider_manager.to_bel()

    if os.path.exists(RESOURCES):
        pybel.to_pickle(sider_graph, DEFAULT_SIDER_PICKLE)

    return sider_graph


def get_drugbank_graph() -> pybel.BELGraph:
    """Get the DrugBank graph."""
    if os.path.exists(DEFAULT_DRUGBANK_PICKLE):
        return pybel.from_pickle(DEFAULT_DRUGBANK_PICKLE)

    drugbank_manager = bio2bel_drugbank.Manager()
    if not drugbank_manager.is_populated():
        drugbank_manager.populate()
    drugbank_graph = drugbank_manager.to_bel()

    if os.path.exists(RESOURCES):
        pybel.to_pickle(drugbank_graph, DEFAULT_DRUGBANK_PICKLE)

    return drugbank_graph


def combine_pubchem_drugbank(
        *,
        mapping_path,
        drugbank_graph_path=None,
        sider_graph_path=None
):
    """
    Combine the SIDER and DrugBank graphs.

    :param mapping_path: the path to tsv file with mappings between pubchemid and drugbankid
    :param drugbank_graph_path: the path to drugbank graph
    :param sider_graph_path: the path to sider graph
    :return: BELGraph
    """
    if drugbank_graph_path is not None:
        drugbank_graph = pybel.from_pickle(drugbank_graph_path)
    else:
        drugbank_graph = get_drugbank_graph()
    if sider_graph_path is not None:
        sider_graph = pybel.from_pickle(sider_graph_path)
    else:
        sider_graph = get_sider_graph()
    drugbank_pubchem_mapping = pd.read_csv(
        mapping_path, sep="\t",
        index_col=False, dtype={'PubchemID': str, 'Smiles': str, 'DrugbankID': str})
    drugbank_pubchem_mapping = drugbank_pubchem_mapping.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=None,
        inplace=False
    )
    drugbank_to_pubchem = {}
    for ind, row in tqdm(drugbank_pubchem_mapping.iterrows(), desc='create pubchem-drugbank mapping dictionary'):
        drugbank_to_pubchem[pybel.dsl.Abundance(
            namespace=DRUGBANK_NAMESPACE,
            name=row['DrugbankName'],
            identifier=row['DrugbankID'])] = pybel.dsl.Abundance(
            namespace=PUBCHEM_NAMESPACE,
            identifier=row['PubchemID'])
    drugbank_relabel = nx.relabel_nodes(drugbank_graph, drugbank_to_pubchem)
    rm_nodes = []
    for node in drugbank_relabel.nodes():
        if node.namespace == DRUGBANK_NAMESPACE:
            rm_nodes.append(node)
    for node in tqdm(rm_nodes, desc='Removing nodes that were not relabeled'):
        drugbank_relabel.remove_node(node)
    full_graph = sider_graph + drugbank_relabel
    full_graph.remove_nodes_from(list(nx.isolates(full_graph)))
    return full_graph


def create_graph_mapping(*, graph_path, mapping_file_path=os.path.join(DEFAULT_MAPPING_PATH)):
    """
    Create graph mapping.

    The method will get a graph, relabel its nodes and map the nodes to their original names.
    :param graph_path: the path to a graph
    :param mapping_file_path: the path to save the node_mapping_df
    :return: a relabeled graph and a dataframe with the node information
    """
    graph = pybel.from_pickle(graph_path)
    relabel_graph = {}
    i = 1
    for node in tqdm(graph.nodes(), desc='Relabel graph nodes'):
        relabel_graph[node] = i
        i += 1
    node_mapping_list = []
    protein_list = []
    for node, node_id in tqdm(relabel_graph.items(), desc='Create mapping dataframe'):
        name = node.name
        if node.namespace == PUBCHEM_NAMESPACE:
            name = cid_to_synonyms(node.identifier)
            if not isinstance(name, str):
                name = name.decode("utf-8")
        if node.namespace == UNIPROT_NAMESPACE:
            protein_list.append(node.identifier)
        node_mapping_list.append((node_id, node.namespace, node.identifier, name))
    node_mapping_df = pd.DataFrame(node_mapping_list, columns=['node_id', 'namespace', 'identifier', 'name'])
    protein_to_gene = get_gene_names(protein_list)
    for protein, gene in tqdm(protein_to_gene.items(), desc='Enrich proteins'):
        node_mapping_df.loc[node_mapping_df['identifier'] == protein, 'name'] = gene
    node_mapping_df.to_csv(mapping_file_path, index=False, sep='\t')
    graph_id = nx.relabel_nodes(graph, relabel_graph)
    return graph_id, relabel_graph


def create_chemicals_mapping_file(
        *,
        drugbank_file,
        mapping_filepath
):
    """
    Create a tsv file containing chemical mapping information.

    The csv file will contain 4 columns: pubchemID, drugbankID, drugbankName and the SMILES.
    :param drugbank_file: to get this file you need to register in drugbank and download full database.xml file
    :param mapping_filepath: the path in which the tsv mapping file will be saved
    :return: a dataframe with the mapping information
    """
    tree = ElementTree.parse(drugbank_file)
    root = tree.getroot()
    ns = '{http://www.drugbank.ca}'
    smiles_template = "{ns}calculated-properties/{ns}property[{ns}kind='SMILES']/{ns}value"
    drugbank_name = []
    drugbank_id = []
    drug_smiles = []
    for i, drug in tqdm(enumerate(root), desc="Getting DrugBank info"):
        assert drug.tag == ns + 'drug'
        if drug.findtext(smiles_template.format(ns=ns)) is None:
            continue
        drugbank_name.append(drug.findtext(ns + "name"))
        drug_smiles.append(drug.findtext(smiles_template.format(ns=ns)))
        drugbank_id.append(drug.findtext(ns + "drugbank-id"))
    pubchem_ids = []
    for smile in tqdm(drug_smiles, desc="Getting PubChemID"):
        pubchem = smiles_to_cid(smile)
        if not isinstance(pubchem, str):
            pubchem = pubchem.decode("utf-8")
        pubchem_ids.append(pubchem)
    mapping_dict = {'PubchemID': pubchem_ids, 'DrugbankID': drugbank_id, 'DrugbankName': drugbank_name,
                    'Smiles': drug_smiles}
    mapping_df = pd.DataFrame(mapping_dict)
    mapping_df.to_csv(mapping_filepath, sep='\t', index=False)
    return mapping_df
