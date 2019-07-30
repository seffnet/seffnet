
"""
Handle pubchem requests.
"""

import urllib
import urllib.request
from urllib.error import HTTPError


def get_result(url):
    """
    Get response from API.

    :param url: API url
    :return: reponse body
    """
    try:
        connection = urllib.request.urlopen(url)
    except HTTPError:
        return ""
    else:
        return connection.read().rstrip()


def cid_to_smiles(cid):
    """
    Get the SMILES for chemicals in PubChem database.

    :param cid: pubchem identifier
    :return: SMILES
    """
    return get_result("http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/property/canonicalSMILES/TXT" % cid)


def smiles_to_cid(smiles):
    """
    Get the chemical pubchem ID from the SMILES.

    :param smiles: the SMILES code of a chemical
    :return: the pubchem identifier
    """
    return get_result("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%s/cids/TXT" % smiles)


def cid_to_synonyms(cid):
    """
    Get the synonymes of chemical in PubChem database.
    :param cid: pubchem identifier
    :return: IUPAC name of the chemical
    """

    return get_result("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/property/IUPACName/TXT" % cid)

