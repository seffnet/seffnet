# -*- coding: utf-8 -*-

"""Pre-processing of Graphs used for NRL models."""

import os

import networkx as nx
import pandas as pd
import pybel
from defusedxml import ElementTree
from tqdm import tqdm

from .constants import (
    DEFAULT_CHEMICALS_MAPPING_PATH, DEFAULT_DRUGBANK_PICKLE, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST,
    DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE, DEFAULT_MAPPING_PATH, DEFAULT_SIDER_PICKLE, PUBCHEM_NAMESPACE, RESOURCES
)
from .get_url_requests import cid_to_synonyms, inchikey_to_cid, cid_to_inchikey


def get_sider_graph(rebuild: bool = False) -> pybel.BELGraph:
    """Get the SIDER graph."""
    if not rebuild and os.path.exists(DEFAULT_SIDER_PICKLE):
        return pybel.from_pickle(DEFAULT_SIDER_PICKLE)

    import bio2bel_sider

    sider_manager = bio2bel_sider.Manager()
    if not sider_manager.is_populated():
        sider_manager.populate()
    sider_graph = sider_manager.to_bel()

    if os.path.exists(RESOURCES):
        pybel.to_pickle(sider_graph, DEFAULT_SIDER_PICKLE)

    return sider_graph


def get_drugbank_graph(rebuild: bool = False, **kwargs) -> pybel.BELGraph:
    """Get the DrugBank graph."""
    if not rebuild and os.path.exists(DEFAULT_DRUGBANK_PICKLE):
        return pybel.from_pickle(DEFAULT_DRUGBANK_PICKLE)

    import bio2bel_drugbank

    drugbank_manager = bio2bel_drugbank.Manager()
    if not drugbank_manager.is_populated():
        drugbank_manager.populate()
    drugbank_graph = drugbank_manager.to_bel(**kwargs)

    if os.path.exists(RESOURCES):
        pybel.to_pickle(drugbank_graph, DEFAULT_DRUGBANK_PICKLE)

    return drugbank_graph


def get_combined_sider_drugbank(
        *,
        rebuild: bool = False,
        drugbank_graph_path=None,
        sider_graph_path=None,
        chemical_mapping=DEFAULT_CHEMICALS_MAPPING_PATH,
):
    """
    Combine the SIDER and DrugBank graphs.

    :param drugbank_graph_path: the path to drugbank graph
    :param sider_graph_path: the path to sider graph
    :return: BELGraph
    """
    if not rebuild and os.path.exists(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE):
        return pybel.from_pickle(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE)
    if type(sider_graph_path) == pybel.struct.graph.BELGraph:
        sider_graph = sider_graph_path
    elif sider_graph_path is not None and os.path.exists(sider_graph_path):
        sider_graph = pybel.from_pickle(sider_graph_path)
    else:
        sider_graph = get_sider_graph()
    if type(drugbank_graph_path) == pybel.struct.graph.BELGraph:
        drugbank_graph = drugbank_graph_path
    elif drugbank_graph_path is not None and os.path.exists(drugbank_graph_path):
        drugbank_graph = pybel.from_pickle(drugbank_graph_path)
    else:
        drugbank_graph = get_drugbank_graph()
    inchi_dict = {}
    if chemical_mapping is not None:
        mapping_df = pd.read_csv(
            chemical_mapping,
            sep="\t",
            dtype={'PubchemID': str, 'InChiKey': str},
            index_col=False,
        )
        for ind, row in mapping_df.iterrows():
            node = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=row['PubchemID'])
            inchi_dict[str(node)] = row['InChiKey'].split('-')[0]
    for node in tqdm(sider_graph.nodes()):
        if node.namespace != 'pubchem.compound':
            continue
        if node in inchi_dict.keys():
            continue
        inchi = cid_to_inchikey(node.identifier)
        if not isinstance(inchi, str):
            inchi = inchi.decode("utf-8")
        inchi_dict[node] = inchi.split('-')[0]
    for node in tqdm(drugbank_graph.nodes()):
        if node.namespace != 'pubchem.compound':
            continue
        if node in inchi_dict.keys():
            continue
        inchi = cid_to_inchikey(node.identifier)
        if not isinstance(inchi, str):
            inchi = inchi.decode("utf-8")
        inchi_dict[node] = inchi.split('-')[0]
    sider_relabeled = nx.relabel_nodes(sider_graph, inchi_dict)
    drugbank_relabeled = nx.relabel_nodes(drugbank_graph, inchi_dict)
    full_graph = sider_relabeled + drugbank_relabeled
    inchi_dict_rev = {v: k for k, v in inchi_dict.items()}
    full_graph_relabel = nx.relabel_nodes(full_graph, inchi_dict_rev)
    if os.path.exists(RESOURCES):
        pybel.to_pickle(full_graph_relabel, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE)
    return full_graph_relabel


def get_mapped_graph(
        *,
        graph_path=DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE,
        rebuild: bool = False,
):
    """
    Create graph mapping.

    The method will get a graph, relabel its nodes and map the nodes to their original names.
    :param graph_path: the path to a graph
    :param mapping_file_path: the path to save the node_mapping_df
    :return: a relabeled graph and a dataframe with the node information
    """
    if not rebuild and os.path.exists(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST):
        return nx.read_edgelist(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST)
    if type(graph_path) == pybel.struct.graph.BELGraph:
        graph = graph_path
    else:
        graph = pybel.from_pickle(graph_path)
    chemical_mapping = None
    if os.path.exists(DEFAULT_CHEMICALS_MAPPING_PATH):
        chemical_mapping = pd.read_csv(
            DEFAULT_CHEMICALS_MAPPING_PATH,
            sep="\t",
            dtype={'PubchemID': str, 'InChiKey': str},
            index_col=False,
        ).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    relabel_graph = {}
    i = 1
    for node in tqdm(graph.nodes(), desc='Relabel graph nodes'):
        relabel_graph[node] = i
        i += 1
    node_mapping_list = []
    for node, node_id in tqdm(relabel_graph.items(), desc='Create mapping dataframe'):
        name = node.name
        if node.namespace == PUBCHEM_NAMESPACE:
            if chemical_mapping.loc[chemical_mapping["PubchemID"] == node.identifier].empty:
                synonyms = cid_to_synonyms(node.identifier)
                if not isinstance(synonyms, str):
                    synonyms = synonyms.decode("utf-8")
                name = synonyms.split('\n')[0]
            else:
                name = chemical_mapping.loc[chemical_mapping['PubchemID'] == node.identifier, 'DrugbankName'].iloc[0]
        node_mapping_list.append((node_id, node.namespace, node.identifier, name))
    node_mapping_df = pd.DataFrame(node_mapping_list, columns=['node_id', 'namespace', 'identifier', 'name'])
    node_mapping_df.to_csv(DEFAULT_MAPPING_PATH, index=False, sep='\t')
    graph_id = nx.relabel_nodes(graph, relabel_graph)
    nx.write_edgelist(graph_id, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST, data=False)
    return graph_id


def get_chemicals_mapping_file(
        *,
        drugbank_file=None,
        mapping_filepath=DEFAULT_CHEMICALS_MAPPING_PATH,
        rebuild: bool = False,
):
    """
    Create a tsv file containing chemical mapping information.

    The csv file will contain 4 columns: pubchemID, drugbankID, drugbankName and the SMILES.
    :param drugbank_file: to get this file you need to register in drugbank and download full database.xml file
    :param mapping_filepath: the path in which the tsv mapping file will be saved
    :return: a dataframe with the mapping information
    """
    if not rebuild and os.path.exists(mapping_filepath):
        return pd.read_csv(
            DEFAULT_CHEMICALS_MAPPING_PATH,
            sep="\t",
            dtype={'PubchemID': str, 'Smiles': str},
            index_col=False,
        )
    tree = ElementTree.parse(drugbank_file)
    root = tree.getroot()
    ns = '{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    smiles_template = "{ns}calculated-properties/{ns}property[{ns}kind='SMILES']/{ns}value"
    pubchem_template = \
        "{ns}external-identifiers/{ns}external-identifier[{ns}resource='PubChem Compound']/{ns}identifier"
    drugbank_name = []
    drugbank_id = []
    drug_inchikey = []
    drug_smiles = []
    pubchem_ids = []
    for i, drug in tqdm(enumerate(root), desc="Getting DrugBank info"):
        assert drug.tag == ns + 'drug'
        if drug.attrib['type'] == "biotech":
            continue
        if drug.findtext(inchikey_template.format(ns=ns)) is None:
            continue
        drugbank_name.append(drug.findtext(ns + "name"))
        inchikey = drug.findtext(inchikey_template.format(ns=ns))
        smiles = drug.findtext(smiles_template.format(ns=ns))
        drug_smiles.append(smiles)
        drug_inchikey.append(inchikey)
        drugbank_id.append(drug.findtext(ns + "drugbank-id"))
        pubchem_id = drug.findtext(pubchem_template.format(ns=ns))
        if pubchem_id is not None:
            pubchem_ids.append(pubchem_id)
        else:
            pubchem_id = inchikey_to_cid(inchikey)
            if not isinstance(pubchem_id, str):
                pubchem_id = pubchem_id.decode("utf-8")
            pubchem_ids.append(pubchem_id)
    mapping_dict = {
        'PubchemID': pubchem_ids, 'DrugbankID': drugbank_id, 'DrugbankName': drugbank_name,
        'InChiKey': drug_inchikey, 'Smiles': drug_smiles
    }
    mapping_df = pd.DataFrame(mapping_dict)
    mapping_df = mapping_df.dropna()
    mapping_df.to_csv(mapping_filepath, sep='\t', index=False)
    return mapping_df
