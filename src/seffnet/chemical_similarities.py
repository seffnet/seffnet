# -*- coding: utf-8 -*-

"""
Chemical similarities calculations.

This file contains functions that calculate similarities between chemicals and produce a chemical similarity BELGraph
Note: to run these the similarity function you need to have RDKit package
"""

import itertools as itt
import logging
import os

import networkx as nx
import pandas as pd
import pybel
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from .constants import (
    DEFAULT_CHEMICALS_MAPPING_PATH, DEFAULT_CHEMSIM_PICKLE, DEFAULT_CHEMSIM_WEIGHTED_PICKLE,
    DEFAULT_CLUSTERED_CHEMICALS, DEFAULT_FULLGRAPH_PICKLE, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE, DEFAULT_GRAPH_PATH,
    DEFAULT_GRAPH_WEIGHTED_PATH, DEFAULT_MAPPING_PATH, PUBCHEM_NAMESPACE, UNIPROT_NAMESPACE,
)
from .get_url_requests import cid_to_smiles, cid_to_synonyms

logger = logging.getLogger(__name__)


def get_smiles(pubchem_ids):
    """
    Get SMILES of a list of chemicals.

    :param pubchem_ids: a list of chemicals as pubchem ID
    :return: a dictionary with pubchemID as key and smiles as value
    """
    pubchem_id_to_smiles = {}
    for pubchem_id in tqdm(pubchem_ids, desc='Getting SMILES'):
        smiles = cid_to_smiles(pubchem_id)
        if not isinstance(smiles, str):
            smiles = smiles.decode("utf-8")
        if smiles is None:
            continue
        pubchem_id_to_smiles[pubchem_id] = smiles
    return pubchem_id_to_smiles


def get_similarity(pubchem_id_to_fingerprint, precision: int = 3):
    """Get the similarities between all pair combinations of chemicals in the list.

    :param pubchem_id_to_fingerprint: a dictionary with pubchem compound ID as key and smiles as value
    :return: a dictionary with the pair chemicals as key and similarity calculation as value
    """
    n_elements = len(pubchem_id_to_fingerprint)
    it = tqdm(
        itt.combinations(pubchem_id_to_fingerprint.items(), 2),
        total=(n_elements * (n_elements - 1) / 2),
        desc='Calculating Similarities',
    )
    return {
        (pubchem_id_1, pubchem_id_2): round(DataStructs.FingerprintSimilarity(mol_1, mol_2), precision)
        for (pubchem_id_1, mol_1), (pubchem_id_2, mol_2) in it
    }


def get_fingerprints(pubchem_id_to_smiles):
    """Create a dictionary containing the fingerprints for every chemical.

    :param pubchem_id_to_smiles: a dictionary with pubchemID as keys and smiles as values
    :return: a dictionary with pubchemID as key and the MACCSkeys fingerprints
    """
    pubchem_id_to_fingerprint = {}
    for pubchem_id, smiles in tqdm(pubchem_id_to_smiles.items(), desc='Getting fingerprints'):
        if not pubchem_id or pd.isna(pubchem_id):
            continue
        if not smiles or pd.isna(smiles):
            logger.debug(f'Missing smiles for {PUBCHEM_NAMESPACE}:{smiles}')
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        pubchem_id_to_fingerprint[pubchem_id] = MACCSkeys.GenMACCSKeys(mol)
    return pubchem_id_to_fingerprint


def parse_chemical_mapping(mapping_file, pubchem_ids):
    """Parse chemical mapping file and create pubchem to smiles dict."""
    chemicals_mapping = pd.read_csv(
        mapping_file,
        sep="\t",
        dtype={'pubchem_id': str, 'smiles': str},
        index_col=False,
    )
    pubchem_id_to_smiles = {}
    new_chemicals = []
    smiles = []
    for pubchem_id in tqdm(pubchem_ids, desc="Getting SMILES"):
        if chemicals_mapping.loc[chemicals_mapping["pubchem_id"] == pubchem_id].empty:
            chemical_smiles = cid_to_smiles(pubchem_id)
            if not isinstance(chemical_smiles, str):
                chemical_smiles = chemical_smiles.decode("utf-8")
            pubchem_id_to_smiles[pubchem_id] = chemical_smiles
            new_chemicals.append(pubchem_id)
            smiles.append(chemical_smiles)
        else:
            pubchem_id_to_smiles[pubchem_id] = chemicals_mapping.loc[chemicals_mapping["pubchem_id"] == pubchem_id,
                                                                     "smiles"].iloc[0]
    new_df = pd.DataFrame({"pubchem_id": new_chemicals, "smiles": smiles})
    chemicals_mapping = chemicals_mapping.append(new_df)
    chemicals_mapping.to_csv(mapping_file, sep='\t', index=False)
    return pubchem_id_to_smiles


def create_clustered_chemsim_graph(
    *,
    pubchem_id_to_fingerprint,
    chemsim_graph,
    weighted: bool = False,
):
    """Create clustered chemsim graph."""
    if weighted:
        similarities = get_similarity(pubchem_id_to_fingerprint)
    clustered_df = cluster_chemicals(rebuild=True, chemicals_dict=pubchem_id_to_fingerprint)
    clusters = clustered_df['Cluster'].unique().tolist()
    for cluster in tqdm(clusters, desc='Creating similarity BELGraph'):
        chemicals = clustered_df.loc[clustered_df.Cluster == cluster]
        if len(chemicals) == 1:
            continue
        for ind, row in chemicals.iterrows():
            for ind1, row1 in chemicals.iterrows():
                if row['PubchemID'] == row1['PubchemID']:
                    continue
                chemical_01 = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=row['PubchemID'])
                chemical_02 = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=row1['PubchemID'])
                if chemsim_graph.has_edge(chemical_01, chemical_02) or chemsim_graph.has_edge(chemical_02,
                                                                                              chemical_01):
                    continue
                chemsim_graph.add_unqualified_edge(chemical_01, chemical_02, 'association')
                if weighted:
                    if (row['PubchemID'], row1['PubchemID']) in similarities:
                        sim = similarities[(row['PubchemID'], row1['PubchemID'])]
                    else:
                        sim = similarities[(row1['PubchemID'], row['PubchemID'])]
                    for iden, edge_d in chemsim_graph[chemical_01][chemical_02].items():
                        chemsim_graph[chemical_01][chemical_02][iden]['weight'] = sim
    return chemsim_graph


def get_similarity_graph(
    *,
    fullgraph=DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE,
    rebuild: bool = False,
    mapping_file=DEFAULT_CHEMICALS_MAPPING_PATH,
    chemsim_graph_path=None,
    clustered: bool = True,
    weighted: bool = False,
    similarity=0.7,
    name='Chemical Similarity Graph',
    version='1.1.0',
    authors='',
    contact='',
    description='',
):
    """
    Create a BELGraph with chemicals as nodes, and similarity as edges.

    :param similarity: the percent in which the chemicals are similar
    :param mapping_file: an existing dataframe with pubchemIDs and Smiles
    """
    if not rebuild and weighted and os.path.exists(DEFAULT_CHEMSIM_WEIGHTED_PICKLE):
        return nx.read_edgelist(DEFAULT_CHEMSIM_WEIGHTED_PICKLE)
    elif not rebuild and os.path.exists(DEFAULT_CHEMSIM_PICKLE):
        return nx.read_edgelist(DEFAULT_CHEMSIM_PICKLE)
    if type(fullgraph) == pybel.struct.graph.BELGraph:
        fullgraph_without_chemsim = fullgraph
    else:
        fullgraph_without_chemsim = pybel.from_pickle(fullgraph)
    pubchem_ids = []
    for node in fullgraph_without_chemsim.nodes():
        if node.namespace != PUBCHEM_NAMESPACE:
            continue
        pubchem_ids.append(node.identifier)

    if os.path.exists(mapping_file):
        pubchem_id_to_smiles = parse_chemical_mapping(mapping_file, pubchem_ids)
    else:
        pubchem_id_to_smiles = get_smiles(pubchem_ids)

    pubchem_id_to_fingerprint = get_fingerprints(pubchem_id_to_smiles)

    chemsim_graph = pybel.BELGraph(name, version, description, authors, contact)

    if clustered:
        chemsim_graph = create_clustered_chemsim_graph(
            pubchem_id_to_fingerprint=pubchem_id_to_fingerprint,
            chemsim_graph=chemsim_graph,
            weighted=weighted
        )
    else:
        similarities = get_similarity(pubchem_id_to_fingerprint)
        for (source_pubchem_id, target_pubchem_id), sim in tqdm(similarities.items(),
                                                                desc='Creating similarity BELGraph'):
            if sim < similarity:
                continue
            node_1 = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=source_pubchem_id)
            node_2 = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=target_pubchem_id)
            chemsim_graph.add_unqualified_edge(node_1, node_2, 'association')
            if weighted:
                for iden, edge_d in chemsim_graph[node_1][node_2].items():
                    chemsim_graph[node_1][node_2][iden]['weight'] = sim
    if chemsim_graph_path is not None:
        pybel.to_pickle(chemsim_graph, chemsim_graph_path)
    elif weighted:
        pybel.to_pickle(chemsim_graph, DEFAULT_CHEMSIM_WEIGHTED_PICKLE)
    else:
        pybel.to_pickle(chemsim_graph, DEFAULT_CHEMSIM_PICKLE)
    return chemsim_graph


def get_combined_graph_similarity(
    *,
    fullgraph_path=DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE,
    chemsim_graph_path=DEFAULT_CHEMSIM_PICKLE,
    mapping_file=DEFAULT_MAPPING_PATH,
    new_graph_path=None,
    pickle_graph_path=DEFAULT_FULLGRAPH_PICKLE,
    rebuild: bool = False,
    weighted: bool = False,
):
    """Combine chemical similarity graph with the fullgraph."""
    if not rebuild and weighted and os.path.exists(DEFAULT_GRAPH_WEIGHTED_PATH):
        return nx.read_edgelist(DEFAULT_GRAPH_WEIGHTED_PATH)
    elif not rebuild and os.path.exists(DEFAULT_GRAPH_PATH):
        return nx.read_edgelist(DEFAULT_GRAPH_PATH)
    if type(fullgraph_path) == pybel.struct.graph.BELGraph:
        fullgraph_without_chemsim = fullgraph_path
    else:
        fullgraph_without_chemsim = pybel.from_pickle(fullgraph_path)
    if type(chemsim_graph_path) == pybel.struct.graph.BELGraph:
        chemsim_graph = chemsim_graph_path
    else:
        chemsim_graph = pybel.from_pickle(chemsim_graph_path)

    mapping_df = pd.read_csv(
        mapping_file,
        sep="\t",
        dtype={'identifier': str, 'node_id': str},
        index_col=False,
    )
    fullgraph_with_chemsim = fullgraph_without_chemsim + chemsim_graph
    pybel.to_pickle(fullgraph_with_chemsim, pickle_graph_path)
    relabel_graph = {}
    for ind, row in mapping_df.iterrows():
        if row['namespace'] == PUBCHEM_NAMESPACE:
            relabel_graph[pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=row['identifier'])] = \
                row['node_id']
        elif row['namespace'] == UNIPROT_NAMESPACE:
            relabel_graph[pybel.dsl.Protein(namespace=UNIPROT_NAMESPACE, identifier=row['identifier'],
                                            name=row['name'])] = row['node_id']
        else:
            relabel_graph[pybel.dsl.Pathology(namespace='umls', identifier=row['identifier'], name=row['name'])] = \
                row['node_id']

    nx.relabel_nodes(fullgraph_with_chemsim, relabel_graph, copy=False)
    if new_graph_path is None:
        if weighted:
            new_graph_path = DEFAULT_GRAPH_WEIGHTED_PATH
        else:
            new_graph_path = DEFAULT_GRAPH_PATH
    if weighted:
        nx.write_weighted_edgelist(fullgraph_with_chemsim, new_graph_path)
    else:
        nx.write_edgelist(fullgraph_with_chemsim, new_graph_path, data=False)
    return fullgraph_with_chemsim


def cluster_chemicals(
    *,
    rebuild: bool = False,
    chemicals_dict,
):
    """Cluster chemicals based on their similarities."""
    if not rebuild and os.path.exists(DEFAULT_CLUSTERED_CHEMICALS):
        return pd.read_csv(
            DEFAULT_CLUSTERED_CHEMICALS,
            sep="\t",
            index_col=False,
            dtype={'PubchemID': str}
        )
    dists = []
    drugs, fps = zip(*chemicals_dict.items())

    nfps = len(chemicals_dict)
    for i in tqdm(range(1, nfps), desc='Calculating distance for clustering'):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    cs = Butina.ClusterData(dists, nfps, 0.3, isDistData=True)
    df = pd.DataFrame(columns=['PubchemID', 'Cluster'])

    i = 1
    for j, cluster in enumerate(cs, start=1):
        for drug in cluster:
            df.loc[i] = [drugs[drug - 1]] + [j]
            i += 1

    df.to_csv(DEFAULT_CLUSTERED_CHEMICALS, sep='\t', index=False)
    return df


def add_new_chemicals(
    *,
    chemicals_list,
    graph=DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE,
    mapping_file=DEFAULT_MAPPING_PATH,
    chemsim_graph_path=DEFAULT_CHEMSIM_PICKLE,
    updated_graph_path=DEFAULT_GRAPH_PATH,
    pickled_graph_path=DEFAULT_FULLGRAPH_PICKLE,
):
    """
    Add new chemicals to a graph.

    :param chemicals_list: a list of pubchem ids
    :param graph: the graph to update
    :param mapping_file: the node mapping file
    :return: Graph, the fullgraph updated and relabeled
    """
    mapping_df = pd.read_csv(
        mapping_file,
        sep="\t",
        index_col=False,
        dtype={'identifier': str, 'node_id': str},
    )

    fullgraph = pybel.from_pickle(graph)
    namespace = []
    name = []
    node_id = []
    new_chemicals = []
    max_node_id = len(fullgraph.nodes())
    for chemical in chemicals_list:
        node = pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=chemical)
        if node in fullgraph.nodes():
            continue
        namespace.append(PUBCHEM_NAMESPACE)
        synonyms = cid_to_synonyms(chemical)
        if not isinstance(synonyms, str):
            synonyms = synonyms.decode("utf-8")
        name.append(synonyms.split('\n')[0])
        max_node_id += 1
        node_id.append(str(max_node_id))
        new_chemicals.append(chemical)
        fullgraph.add_node(node)
    new_nodes = pd.DataFrame({'node_id': node_id, 'namespace': namespace, 'name': name, 'identifier': new_chemicals})
    mapping_df = mapping_df.append(new_nodes)
    mapping_df.to_csv(mapping_file, sep='\t', index=False)
    chemsim_graph = get_similarity_graph(fullgraph=fullgraph, rebuild=True, chemsim_graph_path=chemsim_graph_path)
    return get_combined_graph_similarity(rebuild=True, fullgraph_path=fullgraph, chemsim_graph_path=chemsim_graph,
                                         new_graph_path=updated_graph_path, pickle_graph_path=pickled_graph_path)
