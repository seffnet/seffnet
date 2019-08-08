# -*- coding: utf-8 -*-

"""Utilities for ``se_kge``."""
import os
from typing import Any, Mapping

import networkx as nx
import optuna
import pandas as pd
import pybel
from se_kge.get_url_requests import smiles_to_cid, cid_to_synonyms, get_gene_names
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

