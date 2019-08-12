# -*- coding: utf-8 -*-

"""
Chemical similarities calculations.

This file contains functions that calculate similarities between chemicals and produce a chemical similarity BELGraph
Note: to run these the similarity function you need to have rdkit package
"""

import itertools as itt
import os

import networkx as nx
import pandas as pd
import pybel
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from .constants import (
    PUBCHEM_NAMESPACE, DEFAULT_CHEMICALS_MAPPING_PATH, DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE, DEFAULT_GRAPH_PATH,
    DEFAULT_FULLGRAPH_PICKLE, DEFAULT_CLUSTERED_CHEMICALS,
    DEFAULT_CHEMSIM_PICKLE)
from .get_url_requests import cid_to_smiles


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


def get_similarity(pubchem_id_to_smiles):
    """
    Get the similarities between all pair combinations of chemicals in the list.

    :param pubchem_id_to_smiles: a dictionary with pubchemID as key and smiles as value
    :return: a dictionary with the pair chemicals as key and similarity calculation as value
    """
    fps = get_fingerprints(pubchem_id_to_smiles)
    chem_sim = {
        (pubchem_id_1, pubchem_id_2): DataStructs.FingerprintSimilarity(mol_1, mol_2)
        for (pubchem_id_1, mol_1), (pubchem_id_2, mol_2) in
        tqdm(itt.combinations(fps.items(), 2), desc='Calculating Similarities')
    }
    return chem_sim


def get_fingerprints(pubchem_id_to_smiles):
    """
    Create a dictionary containing the fingerprints for every chemical.

    :param pubchem_id_to_smiles: a dictionary with pubchemID as keys and smiles as values
    :return: a dictionary with pubchemID as key and the MACCSkeys fingerprints
    """
    pubchem_id_to_fingerprint = {}
    for pubchem_id, smiles in tqdm(pubchem_id_to_smiles.items(), desc='Getting fingerprints'):
        mol_from_smile = Chem.MolFromSmiles(smiles)
        if mol_from_smile is None:
            continue
        pubchem_id_to_fingerprint[pubchem_id] = MACCSkeys.GenMACCSKeys(mol_from_smile)
    return pubchem_id_to_fingerprint


def get_similarity_graph(
        *,
        fullgraph=DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE,
        rebuild: bool = False,
        mapping_file=DEFAULT_CHEMICALS_MAPPING_PATH,
        similarity=0.5,
        name='Chemical Similarity Graph',
        version='1.1.0',
        authors='',
        contact='',
        description='',
):
    """
    Create a BELGraph with chemicals as nodes, and similarity as edges.

    :param pubchem_ids: a list of chemicals as pubchem ID
    :param similarity: the percent in which the chemicals are similar
    :param mapping_file: an existing dataframe with pubchemIDs and Smiles
    """
    if not rebuild and os.path.exists(DEFAULT_GRAPH_PATH):
        return nx.read_edgelist(DEFAULT_GRAPH_PATH)
    fullgraph_without_chemsim = pybel.from_pickle(fullgraph)
    pubchem_ids = []
    for node in fullgraph_without_chemsim.nodes():
        if node.namespace != 'pubchem.compound':
            continue
        pubchem_ids.append(node.identifier)

    if os.path.exists(mapping_file):
        chemicals_mapping = pd.read_csv(
            mapping_file,
            sep="\t",
            dtype={'PubchemID': str, 'Smiles': str},
            index_col=False,
        )
        pubchem_id_to_smiles = {}
        for pubchem_id in tqdm(pubchem_ids, desc="Getting SMILES"):
            if chemicals_mapping.loc[chemicals_mapping["PubchemID"] == pubchem_id].empty:
                pubchem_id_to_smiles[pubchem_id] = cid_to_smiles(pubchem_id)
            else:
                pubchem_id_to_smiles[pubchem_id] = chemicals_mapping.loc[chemicals_mapping["PubchemID"] == pubchem_id,
                                                                         "Smiles"].iloc[0]
    else:
        pubchem_id_to_smiles = get_smiles(pubchem_ids)

    similarities = get_similarity(pubchem_id_to_smiles)

    chemsim_graph = pybel.BELGraph(name, version, description, authors, contact)
    for (source_pubchem_id, target_pubchem_id), sim in tqdm(similarities.items(), desc='Creating BELGraph'):
        if sim < similarity:
            continue
        chemsim_graph.add_unqualified_edge(
            pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=source_pubchem_id),
            pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=target_pubchem_id),
            'association',
        )
    fullgraph_with_chemsim = fullgraph_without_chemsim + chemsim_graph
    relabel_graph = {}
    i = 1
    for node in fullgraph_with_chemsim.nodes():
        relabel_graph[node] = i
        i += 1
    fullgraph_with_chemsim_relabeled = nx.relabel_nodes(fullgraph_with_chemsim, relabel_graph)
    nx.write_edgelist(fullgraph_with_chemsim_relabeled, DEFAULT_GRAPH_PATH, data=False)
    pybel.to_pickle(fullgraph_with_chemsim, DEFAULT_FULLGRAPH_PICKLE)
    pybel.to_pickle(chemsim_graph, DEFAULT_CHEMSIM_PICKLE)
    return fullgraph_with_chemsim


def cluster_chemicals(
        *,
        rebuild: bool = False,
        graph=DEFAULT_FULLGRAPH_WITHOUT_CHEMSIM_PICKLE,
        mapping_file=DEFAULT_CHEMICALS_MAPPING_PATH,
):
    ##TODO: refactor and optimize this code
    if not rebuild and os.path.exists(DEFAULT_CLUSTERED_CHEMICALS):
        return pd.read_csv(
            DEFAULT_CLUSTERED_CHEMICALS,
            sep="\t",
            index_col=False,
        )
    fullgraph = pybel.from_pickle(graph)
    chemicals_mapping = pd.read_csv(
        mapping_file,
        sep="\t",
        dtype={'PubchemID': str, 'Smiles': str},
        index_col=False,
    )
    chem_list = []
    for node in fullgraph.nodes():
        if node.namespace != 'pubchem.compound':
            continue
        chem_list.append(node.identifier)
    mols_dict = {}
    for index, row in chemicals_mapping.iterrows():
        if str(row['PubchemID']) not in chem_list:
            continue
        mols_dict[row['PubchemID']] = Chem.MolFromSmiles(row['Smiles'])
    for chem in tqdm(chem_list):
        if chem not in mols_dict.keys():
            smiles = cid_to_smiles(chem)
            if not isinstance(smiles, str):
                smiles = smiles.decode("utf-8")
            if smiles is None:
                continue
            mols_dict[chem] = Chem.MolFromSmiles(smiles)
    fps_drug = {}
    fps = []
    drugs = []
    for drug, mol in tqdm(mols_dict.items()):
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(fp)
        drugs.append(drug)
        fps_drug[drug] = fp
    dists = []
    nfps = len(fps)
    for i in tqdm(range(1, nfps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    cs = Butina.ClusterData(dists, nfps, 0.3, isDistData=True)
    df = pd.DataFrame(columns=['PubchemID', 'Cluster'])
    i = 1
    j = 1
    for cluster in cs:
        for drug in cluster:
            df.loc[i] = [drugs[drug - 1]] + [j]
            i += 1
        j += 1
    df.to_csv(DEFAULT_CLUSTERED_CHEMICALS, sep='\t', index=False)
    return df
