from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import itertools as itt
from tqdm import tqdm_notebook as tqdm
import urllib.request
import urllib
from urllib.error import HTTPError


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
    for chemical in tqdm(chemicals_list):
        smiles = CidToSmiles(chemical)
        if type(smiles) != str:
            smiles = smiles.decode("utf-8")
        smiles_dict[chemical] = smiles
    return smiles_dict
    ms = {}
    for pubchem, smiles in pubchem_to_smiles.items():
        ms[pubchem] = Chem.MolFromSmiles(smiles)
    fps = {}
    for pubchem_id, mol in tqdm(ms.items()):
        if mol == None:
            continue
        fps[pubchem_id] = MACCSkeys.GenMACCSKeys(mol)  # using MACCS
    chem_sim = {
        (pubchem_id_1, pubchem_id_2): DataStructs.FingerprintSimilarity(mol_1, mol_2)
        for (pubchem_id_1, mol_1), (pubchem_id_2, mol_2) in tqdm(itt.combinations(fps.items(), 2))
    }
    return chem_sim
