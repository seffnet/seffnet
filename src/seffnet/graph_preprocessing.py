# -*- coding: utf-8 -*-

"""Pre-processing of Graphs used for NRL models."""

import os

import networkx as nx
import numpy as np
import pandas as pd
import pybel
from defusedxml import ElementTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .constants import (
    DEFAULT_CHEMICALS_MAPPING_PATH, DEFAULT_DRUGBANK_PICKLE, DEFAULT_DRUGBANK_WEIGHTED_PICKLE,
    DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_EDGELIST, DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE,
    DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE, DEFAULT_MAPPING_PATH,
    DEFAULT_POTENCY_MAPPING_PATH, DEFAULT_SIDER_PICKLE, DEFAULT_SIDER_WEIGHTED_PICKLE, PUBCHEM_NAMESPACE, RESOURCES,
    UNIPROT_NAMESPACE
)
from .get_url_requests import cid_to_inchikey, cid_to_smiles, cid_to_synonyms, get_gene_names, inchikey_to_cid


def get_sider_graph(rebuild: bool = False, weighted: bool = False) -> pybel.BELGraph:
    """Get the SIDER graph."""
    if not rebuild and os.path.exists(DEFAULT_SIDER_PICKLE):
        return pybel.from_pickle(DEFAULT_SIDER_PICKLE)

    import bio2bel_sider

    sider_manager = bio2bel_sider.Manager()
    if not sider_manager.is_populated():
        sider_manager.populate()
    sider_graph = sider_manager.to_bel()

    if weighted:
        frequency_df = bio2bel_sider.parser.get_se_frequency_df()
        frequency_dict = {}
        for stitch_flat, stitch_stereo, umls, effect, desc, freq_lb, freq_up, meddra, umls_meddra, name \
                in frequency_df.values:
            pubchem_id = str(abs(int(stitch_flat[3:])) - 100000000)
            freq = (freq_lb + freq_up) / 2
            frequency_dict[(pubchem_id, umls)] = freq
        for source, target in sider_graph.edges():
            for iden, edge_d in sider_graph[source][target].items():
                if edge_d['relation'] == 'increases':
                    sider_graph[source][target][iden]['weight'] = 1.0
                else:
                    if (str(source.identifier), str(target.identifier)) in frequency_dict:
                        sider_graph[source][target][iden]['weight'] = frequency_dict[
                            (source.identifier, target.identifier)]
                    else:
                        sider_graph[source][target][iden]['weight'] = 0.0
        if os.path.exists(RESOURCES):
            pybel.to_pickle(sider_graph, DEFAULT_SIDER_WEIGHTED_PICKLE)
    else:
        if os.path.exists(RESOURCES):
            pybel.to_pickle(sider_graph, DEFAULT_SIDER_PICKLE)

    return sider_graph


def get_drugbank_graph(
        rebuild: bool = False,
        weighted: bool = False,
        potency_filepath=DEFAULT_POTENCY_MAPPING_PATH,
        **kwargs,
) -> pybel.BELGraph:
    """Get the DrugBank graph."""
    if not rebuild and os.path.exists(DEFAULT_DRUGBANK_WEIGHTED_PICKLE) and weighted:
        return pybel.from_pickle(DEFAULT_DRUGBANK_WEIGHTED_PICKLE)
    elif not rebuild and os.path.exists(DEFAULT_DRUGBANK_PICKLE):
        return pybel.from_pickle(DEFAULT_DRUGBANK_PICKLE)

    import bio2bel_drugbank

    drugbank_manager = bio2bel_drugbank.Manager()
    if not drugbank_manager.is_populated():
        drugbank_manager.populate()
    drugbank_graph = drugbank_manager.to_bel(**kwargs)

    if weighted:
        weighted_drugbank_graph = get_weighted_drugbank_graph(
            drugbank_graph=drugbank_graph,
            potency_filepath=potency_filepath
        )
        if os.path.exists(RESOURCES):
            pybel.to_pickle(weighted_drugbank_graph, DEFAULT_DRUGBANK_WEIGHTED_PICKLE)
        return weighted_drugbank_graph
    else:
        if os.path.exists(RESOURCES):
            pybel.to_pickle(drugbank_graph, DEFAULT_DRUGBANK_PICKLE)
        return drugbank_graph


def get_weighted_drugbank_graph(
        *,
        drugbank_graph,
        potency_filepath,
):
    """Get the weighted DrugBank graph."""
    potency_mapping = pd.read_csv(
        potency_filepath,
        sep='\t',
        index_col=False,
        dtype={'chemical_pubchem_id': str}
    )
    edge_weights = {
        (chemical_pubchem, target_uniprot): normalized_pchembl
        for chemical_pubchem,
        chemical_chembl,
        target_uniprot,
        target_chembl,
        pchembl,
        normalized_pchembl in potency_mapping.values
    }
    for source, target in drugbank_graph.edges():
        for iden, edge_d in drugbank_graph[source][target].items():
            if (str(source.identifier), str(target.identifier)) in edge_weights:
                drugbank_graph[source][target][iden]['weight'] = edge_weights[str(source.identifier),
                                                                              str(target.identifier)]
            else:
                drugbank_graph[source][target][iden]['weight'] = 0.0
    return drugbank_graph


def get_combined_sider_drugbank(
        *,
        rebuild: bool = False,
        drugbank_graph_path=None,
        sider_graph_path=None,
        weighted: bool = False,
        chemical_mapping=DEFAULT_CHEMICALS_MAPPING_PATH,
):
    """
    Combine the SIDER and DrugBank graphs.

    :param drugbank_graph_path: the path to drugbank graph
    :param sider_graph_path: the path to sider graph
    :return: BELGraph
    """
    if not rebuild and weighted and os.path.exists(DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE):
        return pybel.from_pickle(DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE)
    elif not rebuild and os.path.exists(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE):
        return pybel.from_pickle(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE)
    if type(sider_graph_path) == pybel.struct.graph.BELGraph:
        sider_graph = sider_graph_path
    elif sider_graph_path is not None and os.path.exists(sider_graph_path):
        sider_graph = pybel.from_pickle(sider_graph_path)
    else:
        sider_graph = get_sider_graph(rebuild=rebuild, weighted=weighted)
    if type(drugbank_graph_path) == pybel.struct.graph.BELGraph:
        drugbank_graph = drugbank_graph_path
    elif drugbank_graph_path is not None and os.path.exists(drugbank_graph_path):
        drugbank_graph = pybel.from_pickle(drugbank_graph_path)
    else:
        drugbank_graph = get_drugbank_graph(rebuild=rebuild, weighted=weighted, drug_namespace='pubchem.compound')
    smiles_dict = {}
    if chemical_mapping is not None:
        mapping_df = pd.read_csv(
            chemical_mapping,
            sep="\t",
            dtype={'pubchem_id': str, 'smiles': str},
            index_col=False,
        )
    else:
        mapping_df = None
    for node in tqdm(sider_graph.nodes(), desc='get sider chemicals smiles'):
        if node.namespace != 'pubchem.compound':
            continue
        if node.identifier in mapping_df.values:
            smiles = mapping_df.loc[mapping_df['pubchem_id'] == node.identifier, 'smiles'].iloc[0]
        else:
            smiles = cid_to_smiles(node.identifier)
            if not isinstance(smiles, str):
                smiles = smiles.decode("utf-8")
        smiles_dict[node] = smiles
    for node in tqdm(drugbank_graph.nodes(), desc='get drugbank chemicals smiles'):
        if node.namespace != 'pubchem.compound':
            continue
        if node in smiles_dict.keys():
            continue
        if node.identifier in mapping_df.values:
            smiles = mapping_df.loc[mapping_df['pubchem_id'] == node.identifier, 'smiles'].iloc[0]
        else:
            smiles = cid_to_smiles(node.identifier)
            if not isinstance(smiles, str):
                smiles = smiles.decode("utf-8")
        smiles_dict[node] = smiles
    sider_relabeled = nx.relabel_nodes(sider_graph, smiles_dict)
    drugbank_relabeled = nx.relabel_nodes(drugbank_graph, smiles_dict)
    full_graph = sider_relabeled + drugbank_relabeled
    smiles_dict_rev = {v: k for k, v in smiles_dict.items()}
    full_graph_relabel = nx.relabel_nodes(full_graph, smiles_dict_rev)
    if weighted:
        if os.path.exists(RESOURCES):
            pybel.to_pickle(full_graph_relabel, DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE)
    else:
        if os.path.exists(RESOURCES):
            pybel.to_pickle(full_graph_relabel, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE)
    return full_graph_relabel


def get_mapped_graph(
        *,
        graph_path=None,
        mapping_path=DEFAULT_MAPPING_PATH,
        edgelist_path=None,
        rebuild: bool = False,
        weighted: bool = False,
):
    """
    Create graph mapping.

    The method will get a graph, relabel its nodes and map the nodes to their original names.
    :param graph_path: the path to a graph
    :param mapping_file_path: the path to save the node_mapping_df
    :return: a relabeled graph and a dataframe with the node information
    """
    if not rebuild and weighted and os.path.exists(DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_EDGELIST):
        return nx.read_edgelist(DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_EDGELIST)
    elif not rebuild and os.path.exists(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST):
        return nx.read_edgelist(DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST)
    if graph_path is None:
        if weighted:
            graph_path = DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_PICKLE
        else:
            graph_path = DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE
    if edgelist_path is None:
        if weighted:
            edgelist_path = DEFAULT_FULLGRAPH_WEIGHTED_WITHOUT_CHEMSIM_EDGELIST
        else:
            edgelist_path = DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_EDGELIST
    if type(graph_path) == pybel.struct.graph.BELGraph:
        graph = graph_path
    else:
        graph = pybel.from_pickle(graph_path)
    chemicals_info = {}
    if os.path.exists(DEFAULT_CHEMICALS_MAPPING_PATH):
        chemical_mapping = pd.read_csv(
            DEFAULT_CHEMICALS_MAPPING_PATH,
            sep='\t',
            dtype={'pubchem_id': str, 'smiles': str},
            index_col=False,
        )
        for pubchem_id, drugbank_id, chembl_id, name, drug_group, smiles, inchikey in chemical_mapping.values:
            if pubchem_id is not None:
                chemicals_info[pubchem_id] = dict(
                    name=name,
                    drug_group=drug_group,
                )
    relabel_graph = {}
    i = 1
    for node in tqdm(graph.nodes(), desc='Relabel graph nodes'):
        relabel_graph[node] = i
        i += 1
    node_mapping_list = []
    for node, node_id in tqdm(relabel_graph.items(), desc='Create mapping dataframe'):
        name = node.name
        if node.namespace == PUBCHEM_NAMESPACE:
            if node.identifier not in chemicals_info.keys():
                synonyms = cid_to_synonyms(node.identifier)
                if not isinstance(synonyms, str):
                    synonyms = synonyms.decode("utf-8")
                name = synonyms.split('\n')[0]
                entity_type = 'chemical'
            else:
                name = chemicals_info[node.identifier]['name']
                entity_type = chemicals_info[node.identifier]['drug_group'] + ' drug'
        elif node.namespace == UNIPROT_NAMESPACE:
            entity_type = 'target'
        else:
            entity_type = 'phenotype'
        node_mapping_list.append((node_id, node.namespace, node.identifier, name, entity_type))
    node_mapping_df = pd.DataFrame(
        node_mapping_list,
        columns=['node_id', 'namespace', 'identifier', 'name', 'type']
    )
    node_mapping_df.to_csv(mapping_path, index=False, sep='\t')
    graph_id = nx.relabel_nodes(graph, relabel_graph)
    if weighted:
        nx.write_weighted_edgelist(graph_id, edgelist_path)
    else:
        nx.write_edgelist(graph_id, edgelist_path, data=False)
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
            dtype={'pubchem_id': str, 'smiles': str},
            index_col=False,
        )
    tree = ElementTree.parse(drugbank_file)
    root = tree.getroot()
    ns = '{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    pubchem_template = \
        "{ns}external-identifiers/{ns}external-identifier[{ns}resource='PubChem Compound']/{ns}identifier"
    chembl_template = \
        "{ns}external-identifiers/{ns}external-identifier[{ns}resource='ChEMBL']/{ns}identifier"
    drug_group_template = "{ns}groups/{ns}group"
    mapping_list = []
    for i, drug in tqdm(enumerate(root), desc="Getting DrugBank info"):
        assert drug.tag == ns + 'drug'
        if drug.attrib['type'] == "biotech":
            continue
        name = drug.findtext(ns + "name")
        drugbank_id = drug.findtext(ns + "drugbank-id")
        pubchem_id = drug.findtext(pubchem_template.format(ns=ns))
        chembl_id = drug.findtext(chembl_template.format(ns=ns))
        inchikey = drug.findtext(inchikey_template.format(ns=ns))
        if pubchem_id is None:
            pubchem_id = inchikey_to_cid(inchikey)
            if not isinstance(pubchem_id, str):
                pubchem_id = pubchem_id.decode("utf-8")
        if '\n' in pubchem_id:
            pubchem_id = pubchem_id.split('\n')[0]
        smiles = cid_to_smiles(pubchem_id)
        if not isinstance(smiles, str):
            smiles = smiles.decode("utf-8")
        drug_group = drug.findtext(drug_group_template.format(ns=ns))
        mapping_list.append((pubchem_id, drugbank_id, chembl_id, name, drug_group, smiles, inchikey))
    mapping_df = pd.DataFrame(
        mapping_list,
        columns=['pubchem_id', 'drugbank_id', 'chembl_id', 'name', 'drug_group', 'smiles', 'inchikey']
    )
    mapping_df.to_csv(mapping_filepath, sep='\t', index=False)
    return mapping_df


def map_chemical_target_potency(
        *,
        graph_path=DEFAULT_DRUGBANK_PICKLE,
        chemicals_mapping=DEFAULT_CHEMICALS_MAPPING_PATH,
        mapping_filepath=DEFAULT_POTENCY_MAPPING_PATH,
):
    """Extract chemical to target potency from chembl and normalize the values."""
    from chembl_webresource_client.new_client import new_client

    graph = nx.DiGraph(nx.read_gpickle(graph_path))
    chemicals_info = {}
    if os.path.exists(chemicals_mapping):
        chemical_mapping = pd.read_csv(
            DEFAULT_CHEMICALS_MAPPING_PATH,
            sep="\t",
            dtype={'pubchem_id': str, 'smiles': str},
            index_col=False,
        )
        for pubchem_id, drugbank_id, chembl_id, name, drug_group, smiles, inchikey in \
                tqdm(chemical_mapping.values, desc='Parse chemical mapping'):
            if pubchem_id is not None:
                chemicals_info[pubchem_id] = dict(
                    chembl_id=chembl_id,
                    inchikey=inchikey,
                )
    mapping_list = []
    for edge in tqdm(graph.edges(), desc='Find pchembl values of interactions'):
        if edge[0].identifier not in chemicals_info:
            inchikey = cid_to_inchikey(edge[0].identifier)
            if not isinstance(inchikey, str):
                inchikey = inchikey.decode("utf-8")
            try:
                molecule = new_client.molecule
                m1 = molecule.get(inchikey)
                chemical_chembl = m1['molecule_chembl_id']
            except Exception:
                chemical_chembl = None
        else:
            chemical_chembl = chemicals_info[edge[0].identifier]['chembl_id']
        try:
            target = get_gene_names([edge[1].identifier], to_id='CHEMBL_ID')
            target_chembl = target[edge[1].identifier]
        except Exception:
            target_chembl = None
        if target_chembl is None or chemical_chembl is None:
            continue
        activities = new_client.activity
        results = activities.filter(molecule_chembl_id=chemical_chembl, target_chembl_id=target_chembl,
                                    pchembl_value__isnull=False)
        pchembls = np.array([float(result['pchembl_value']) for result in results])
        if pchembls.size:
            avg_pchembl = np.mean(pchembls)
        else:
            avg_pchembl = 0
        mapping_list.append(
            [edge[0].identifier,
             chemical_chembl,
             edge[1].identifier,
             target_chembl,
             round(avg_pchembl, 3)]
        )
    mapping_df = pd.DataFrame(
        mapping_list,
        columns=['chemical_pubchem_id', 'chemical_chembl_id', 'target_uniprot_id', 'target_chembl_id', 'pchembl'])
    scaler = MinMaxScaler()
    pchembl_scaled = scaler.fit_transform(mapping_df[['pchembl']].values.astype(float))
    mapping_df['normalize_pchembl'] = pchembl_scaled
    mapping_df.to_csv(mapping_filepath, sep='\t', index=False)
    return mapping_df
