# -*- coding: utf-8 -*-

import itertools as itt
import urllib
import urllib.request
from urllib.error import HTTPError

import pybel
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from tqdm import tqdm

"""
This script contains functions that calculate similarities between chemicals and produce a chemical similarity BELGraph 
Note: to run these the similarity function you need to have rdkit package
"""


def getresult(url):
    """
    connect with API to get results
    :param url: the API url
    :return: connection
    """
    try:
        connection = urllib.request.urlopen(url)
    except HTTPError:
        return ""
    else:
        return connection.read().rstrip()


def CidToSmiles(cid):
    """
    Gets the SMILES for chemicals in PubChem database
    :param cid: pubchem ID
    :return: SMILES
    """
    return getresult("http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/property/canonicalSMILES/TXT" % cid)


def SmilesToCid(smiles):
    """
    gets the chemical pubchem ID from the SMILES
    :param smiles: the SMILES code of a chemical
    :return: the pubchem ID
    """
    return getresult("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%s/cids/TXT" % smiles)


def get_similarity(chemicals_list):
    """
    gets the similarities between all pair combinations of chemicals in the list
    :param chemicals_list: a list of chemicals as pubchem ID
    :return: a dictionary with the pair chemicals as key and similarity calculation as value
    """
    smiles_dict = {}
    for chemical in tqdm(chemicals_list, desc='Getting SMILES'):
        smiles = CidToSmiles(chemical)
        if type(smiles) != str:
            smiles = smiles.decode("utf-8")
        smiles_dict[chemical] = smiles
    fps = get_fingerprints(smiles_dict)
    chem_sim = {
        (pubchem_id_1, pubchem_id_2): DataStructs.FingerprintSimilarity(mol_1, mol_2)
        for (pubchem_id_1, mol_1), (pubchem_id_2, mol_2) in
        tqdm(itt.combinations(fps.items(), 2), desc='Calculating Similarities')
    }
    return chem_sim


def get_fingerprints(chemicals_dict):
    """ creates a dictionary containing the fingerprints for every chemical
    :param chemicals_dict: a dictionary with pubchemID as keys and smiles as values
    :return: a dictionary with pubchemID as key and the MACCSkeys fingerprints
    """
    ms = {}
    for pubchem, smiles in tqdm(chemicals_dict.items(), desc='Getting fingerprints'):
        ms[pubchem] = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    return ms


def create_similarity_graph(chemicals_list, name='', version='1.1.0', authors='', contact='', description=''):
    """ Creates a BELGraph with chemicals as nodes, and similarity as edges
    :param chemicals_list: a list of chemicals as pubchem ID
    :return: BELGraph"""
    chem_sim = get_similarity(chemicals_list)
    chem_sim_graph = pybel.BELGraph(name, version, description, authors, contact)
    for (pubchem_1, pubchem_2), sim in tqdm(chem_sim.items(), desc='Creating BELGraph'):
        if sim < 0.5:
            continue
        chem_sim_graph.add_unqualified_edge(pybel.dsl.Abundance(namespace='pubchem', name=pubchem_1),
                                            pybel.dsl.Abundance(namespace='pubchem', name=pubchem_2), 'association')
    return chem_sim_graph
