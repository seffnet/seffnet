# -*- coding: utf-8 -*-

"""Utilities for ``se_kge``."""

from typing import Any, Mapping

import optuna
import pandas as pd
from se_kge.get_url_requests import smiles_to_cid
from tqdm import tqdm
import xml.etree.ElementTree as ET


def study_to_json(study: optuna.Study) -> Mapping[str, Any]:
    """Serialize a study to JSON."""
    return {
        'n_trials': len(study.trials),
        'name': study.study_name,
        'id': study.study_id,
        'start': study.user_attrs['Date'],
        'best': {
            'mcc': study.best_trial.user_attrs['mcc'],
            'accuracy': study.best_trial.user_attrs['accuracy'],
            'auc_roc': study.best_trial.user_attrs['auc_roc'],
            'auc_pr': study.best_trial.user_attrs['auc_pr'],
            'f1': study.best_trial.user_attrs['f1'],
            'method': study.best_trial.user_attrs['method'],
            'params': study.best_params,
            'trial': study.best_trial.number,
            'value': study.best_value,
        },
    }


def create_chemicals_mapping_file(drugbank_file, mapping_filepath):
    """
    Create a tsv file containing chemical mapping information.
    The csv file will contain 4 columns: pubchemID, drugbankID, drugbankName and the SMILES.

    :param drugbank_file: to get this file you need to register in drugbank and download full database.xml file
    :param mapping_filepath: the path in which the tsv mapping file will be saved
    :return: a dataframe with the mapping information
    """
    tree = ET.parse(drugbank_file)
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
    mapping_dict = {'PubchemID': pubchem_ids, 'DrugbankID': drugbank_id, 'DrugbankName': drugbank_name, 'Smiles': drug_smiles}
    mapping_df = pd.DataFrame(mapping_dict)
    mapping_df.to_csv(mapping_filepath, sep='\t', index=False)
    return mapping_df
