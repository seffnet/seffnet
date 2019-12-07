# -*- coding: utf-8 -*-

"""Chemical prediction."""

import logging
from functools import lru_cache
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from .chemical_similarities import get_fingerprints
from .constants import PUBCHEM_NAMESPACE
from .find_relations import Predictor

__all__ = [
    'ChemicalPredictor',
]

logger = logging.getLogger(__name__)


class ChemicalPredictor:
    """A predictor that can calculate new chemicals."""

    predictor: Predictor
    # A mapping of pubchem compound id to RDKit molecule

    fingerprints: Mapping[str, Any]

    def __init__(
        self,
        predictor: Predictor,
        chemical_mapping_path: str
    ):  # noqa: D107
        self.predictor = predictor
        chemical_mapping_df = pd.read_csv(chemical_mapping_path, sep='\t', dtype={'pubchem_id': str, 'smiles': str})
        self.pubchem_id_to_smiles = dict(chemical_mapping_df[['pubchem_id', 'smiles']].values)
        self.fingerprints = get_fingerprints(self.pubchem_id_to_smiles)

    @lru_cache(maxsize=None)
    def embed_smiles(self, smiles: str) -> np.ndarray:
        """Embed a chemical using weighted embeddings of other chemicals by tanimoto similarity of MAACS keys."""
        mol = Chem.MolFromSmiles(smiles)
        return self._embed_mol(mol)

    def _embed_inchi(self, inchi: str) -> np.ndarray:
        mol = Chem.MolFromInchi(inchi)
        return self._embed_mol(mol)

    def _embed_mol(self, mol, precision: int = 3) -> np.ndarray:
        fp = MACCSkeys.GenMACCSKeys(mol)
        weighted_embeddings = []
        failed = 0
        for target_pubchem_id, target_fp in self.fingerprints.items():
            try:
                target_id = self.predictor.node_curie_to_id[PUBCHEM_NAMESPACE, target_pubchem_id]
            except KeyError:
                logger.warning(f'Could not look up {PUBCHEM_NAMESPACE}:{target_pubchem_id}')
                failed += 1
                continue
            embedding = self.predictor.embeddings[target_id]
            similarity = round(DataStructs.FingerprintSimilarity(fp, target_fp), precision)
            weighted_embeddings.append(embedding * similarity)
        if failed:
            logger.warning(f'Could not look up node id for {failed}/{len(self.fingerprints)} chemicals')
        return np.array(weighted_embeddings).mean(axis=0)

    def find_smiles_relations(
        self,
        smiles: str,
        namespace: Optional[str] = None,
        results_type: Optional[str] = None,
        k: Optional[int] = 30,
    ):
        """Embed a chemical by its SMILES."""
        source_id = f'smiles:{smiles}'
        source_vector = self.embed_smiles(smiles)
        relations_results = self.predictor._find_relations_helper(
            source_id=source_id,
            source_vector=source_vector,
            namespace=namespace,
        )
        return self.predictor._handle_relations_results(
            relations_results=relations_results,
            results_type=results_type,
            k=k,
            node_info=dict(namespace='smiles', identifier=smiles, entity_type='drug')
        )
