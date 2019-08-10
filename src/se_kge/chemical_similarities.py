# -*- coding: utf-8 -*-

"""
Chemical similarities calculations.

This file contains functions that calculate similarities between chemicals and produce a chemical similarity BELGraph
Note: to run these the similarity function you need to have rdkit package
"""

import itertools as itt
import os
from collections import Iterable, Mapping
from typing import Any

import pandas as pd
import pybel
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from tqdm import tqdm

from .constants import PUBCHEM_NAMESPACE, RESOURCES
from .get_url_requests import cid_to_smiles


def get_smiles(pubchem_ids: Iterable[str]) -> Mapping[str, str]:
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


def get_similarity(pubchem_id_to_smiles: Mapping[str, str]) -> Mapping[str, float]:
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


def get_fingerprints(pubchem_id_to_smiles: Mapping[str, str]) -> Mapping[str, Any]:
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


def create_similarity_graph(
        pubchem_ids: Iterable[str],
        mapping_file=os.path.abspath(os.path.join(
            RESOURCES,
            "mapping",
            "drugbank_pubchem_mapping.tsv")
        ),
        similarity=0.5,
        name='',
        version='1.1.0',
        authors='',
        contact='',
        description='',
) -> pybel.BELGraph:
    """
    Create a BELGraph with chemicals as nodes, and similarity as edges.

    :param pubchem_ids: a list of chemicals as pubchem ID
    :param similarity: the percent in which the chemicals are similar
    :param mapping_file: an existing dataframe with pubchemIDs and Smiles
    """
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

    bel_graph = pybel.BELGraph(name, version, description, authors, contact)
    for (source_pubchem_id, target_pubchem_id), sim in tqdm(similarities.items(), desc='Creating BELGraph'):
        if sim < similarity:
            continue
        bel_graph.add_unqualified_edge(
            pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=source_pubchem_id),
            pybel.dsl.Abundance(namespace=PUBCHEM_NAMESPACE, identifier=target_pubchem_id),
            'association',
        )
    return bel_graph
