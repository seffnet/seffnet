# -*- coding: utf-8 -*-

"""
Chemical similarities calculations.

This file contains functions that calculate similarities between chemicals and produce a chemical similarity BELGraph
Note: to run these the similarity function you need to have rdkit package
"""

import itertools as itt

import pandas as pd
import pybel
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from tqdm import tqdm

from se_kge.get_url_requests import cid_to_smiles


def get_smiles(chemicals_list):
    """
    Get SMILES of a list of chemicals.

    :param chemicals_list: a list of chemicals as pubchem ID
    :return: a dictionary with pubchemID as key and smiles as value
    """
    smiles_dict = {}
    for chemical in tqdm(chemicals_list, desc='Getting SMILES'):
        smiles = cid_to_smiles(chemical)
        if type(smiles) != str:
            smiles = smiles.decode("utf-8")
        if smiles is None:
            continue
        smiles_dict[chemical] = smiles
    return smiles_dict

def get_similarity(smiles_dict):
    """
    Get the similarities between all pair combinations of chemicals in the list.

    :param smiles_dict: a dictionary with pubchemID as key and smiles as value
    :return: a dictionary with the pair chemicals as key and similarity calculation as value
    """
    fps = get_fingerprints(smiles_dict)
    chem_sim = {
        (pubchem_id_1, pubchem_id_2): DataStructs.FingerprintSimilarity(mol_1, mol_2)
        for (pubchem_id_1, mol_1), (pubchem_id_2, mol_2) in
        tqdm(itt.combinations(fps.items(), 2), desc='Calculating Similarities')
    }
    return chem_sim


def get_fingerprints(chemicals_dict):
    """
    Create a dictionary containing the fingerprints for every chemical.

    :param chemicals_dict: a dictionary with pubchemID as keys and smiles as values
    :return: a dictionary with pubchemID as key and the MACCSkeys fingerprints
    """
    ms = {}
    for pubchem, smiles in tqdm(chemicals_dict.items(), desc='Getting fingerprints'):
        mol_from_smile = Chem.MolFromSmiles(smiles)
        if mol_from_smile is None:
            continue
        ms[pubchem] = MACCSkeys.GenMACCSKeys(mol_from_smile)
    return ms


def create_similarity_graph(chemicals_list, mapping_file=None, similarity=0.5, name='', version='1.1.0', authors='', contact='', description=''):
    """
    Create a BELGraph with chemicals as nodes, and similarity as edges.

    :param chemicals_list: a list of chemicals as pubchem ID
    :param similarity: the percent in which the chemicals are similar
    :param mapping_file: an existing dataframe with pubchemIDs and Smiles
    :return: BELGraph
    """
    if mapping_file is None:
        smiles_dict = get_smiles(chemicals_list)
    else:
        chemicals_mapping = pd.read_csv(mapping_file, sep=",", dtype={'PubchemID':str, 'Smiles':str}, index_col=False)
        smiles_dict = {}
        for chemical in tqdm(chemicals_list, desc="Getting SMILES"):
            if chemicals_mapping.loc[chemicals_mapping["PubchemID"] == chemical].empty:
                smiles_dict[chemical] = cid_to_smiles(chemical)
            else:
                smiles_dict[chemical] = chemicals_mapping.loc[chemicals_mapping["PubchemID"] == chemical, "Smiles"].iloc[0]
    chem_sim = get_similarity(smiles_dict)
    chem_sim_graph = pybel.BELGraph(name, version, description, authors, contact)
    for (pubchem_1, pubchem_2), sim in tqdm(chem_sim.items(), desc='Creating BELGraph'):
        if sim < similarity:
            continue
        chem_sim_graph.add_unqualified_edge(pybel.dsl.Abundance(namespace='pubchem', identifier=pubchem_1),
                                            pybel.dsl.Abundance(namespace='pubchem', identifier=pubchem_2),
                                            'association')
    return chem_sim_graph
