# -*- coding: utf-8 -*-

"""Find new relations between entities.

This file contains functions that find predicted relations from a logistic regression model given model embeddings
The model and embeddings are trained and created from a graph containing drugs, targets and side effects.
The graph used contained nodeIDs that can be mapped using a tsv file
"""

from dataclasses import dataclass
from operator import itemgetter
from typing import Any, List, Mapping, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from bionev.utils import load_embedding
from sklearn.linear_model import LogisticRegression

from .constants import RESULTS_TYPE_TO_NAMESPACE

__all__ = [
    'Embeddings',
    'Predictor',
]

NodeInfo = Mapping[str, str]
Embeddings = Mapping[str, np.ndarray]
RelationsResults = Tuple[List[NodeInfo], List[np.ndarray], List[bool]]


class MissingCurie(ValueError):
    """Raised when a CURIE can't be found."""


def _load_embedding(path: str) -> Embeddings:
    """Load an embedding then fix its data types."""
    rv = load_embedding(path)
    return {
        str(node_id): np.array(node_vector)
        for node_id, node_vector in rv.items()
    }


@dataclass
class Predictor:
    """Class for making predictions."""

    model: LogisticRegression
    embeddings: Embeddings
    node_id_to_info: Mapping[str, NodeInfo]
    node_curie_to_id: Mapping[Tuple[str, str], str]
    node_name_to_id: Mapping[str, str]

    graph: Optional[nx.Graph] = None
    positive_control: bool = True
    #: The precision at which results are reported
    precision: int = 5

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str,
        embeddings_path: str,
        mapping_path: str,
        graph_path: Optional[str] = None,
        positive_control: Optional[bool] = True,
    ) -> 'Predictor':
        """Return the predictor for embeddings."""
        model = joblib.load(model_path)
        mapping = pd.read_csv(mapping_path, sep='\t', dtype={'node_id': str})

        node_id_to_info = {}
        node_curie_to_id = {}
        node_name_to_id = {}
        for node_id, namespace, identifier, name, entity_type in mapping.values:
            node_id_to_info[node_id] = dict(
                node_id=node_id,
                namespace=namespace,
                identifier=identifier,
                name=name,
                entity_type=entity_type,
            )
            node_curie_to_id[namespace, identifier] = node_id
            node_name_to_id[name] = node_id

        embeddings = _load_embedding(embeddings_path)
        graph = graph_path and nx.read_edgelist(graph_path)
        return cls(
            model=model,
            graph=graph,
            embeddings=embeddings,
            positive_control=positive_control,
            node_id_to_info=node_id_to_info,
            node_curie_to_id=node_curie_to_id,
            node_name_to_id=node_name_to_id,
        )

    def find_new_relations(
        self,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        node_curie: Optional[str] = None,
        results_type: Optional[str] = None,
        k: Optional[int] = 30,
    ) -> Optional[Mapping[str, Any]]:
        """Find new relations to specific entity.

        Get all the relations of specific entity_type (if chosen) or all types (if None).
        Finds their probabilities from the saved_model, and return the top k predictions.

        :param node_id: the internal identifier of the node in the model
        :param node_name: the entity we want to find predictions with
        :param node_curie: the CURIE (namespace:identifier) of the entity we want to find predictions with
        :param results_type: can be 'phenotype', 'chemical', 'target', or None
        :param k: the amount of relations we want to find for the entity
        :return: a list of tuples containing the predicted entities and their probabilities
        :raises: MissingCurie
        """
        node_id = self._lookup_node(node_id=node_id, node_curie=node_curie, node_name=node_name)
        if node_id is None:
            raise MissingCurie('The curie you input does not exist.')

        node_info = self._get_entity_json(node_id)

        namespace = RESULTS_TYPE_TO_NAMESPACE.get(results_type)
        relations_results = self._find_relations_helper(
            source_id=node_id,
            source_vector=self.embeddings[node_id],
            namespace=namespace,
        )

        return self._handle_relations_results(
            relations_results=relations_results,
            k=k,
            results_type=results_type,
            node_info=node_info,
        )

    def _handle_relations_results(
        self,
        *,
        relations_results: RelationsResults,
        k: Optional[int],
        results_type: Optional[str],
        node_info,
    ):
        node_list, relations_list, relation_novelties = relations_results
        prediction_list = self.get_probabilities(
            nodes=node_list,
            relations=relations_list,
            relation_novelties=relation_novelties,
            k=k,
        )
        return {
            'query': {
                'entity': node_info,
                'k': k,
                'type': results_type,
            },
            'predictions': prediction_list,
        }

    def _lookup_node_id_by_name(self, entity_name: str) -> str:
        return self.node_name_to_id.get(entity_name)

    def _lookup_node_id_by_curie(self, entity_curie: str) -> str:
        namespace, identifier = entity_curie.split(':', 1)
        return self.node_curie_to_id.get((namespace, identifier))

    def _predict_helper(self, q):
        return self.model.predict_proba(q)[:, 0]

    def _get_entity_json(self, node_id: str) -> NodeInfo:
        return self.node_id_to_info.get(node_id)

    def _lookup_node(
        self,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        node_curie: Optional[str] = None,
    ) -> str:
        if node_id is not None:
            return node_id
        elif node_name is not None:
            return self._lookup_node_id_by_name(node_name)
        elif node_curie is not None:
            return self._lookup_node_id_by_curie(node_curie)
        else:
            raise ValueError("You need to provide information about the entity (node_id, entity_id, or entity_name)")

    def find_new_relation(
        self,
        *,
        source_id: Optional[str] = None,
        source_curie: Optional[str] = None,
        source_name: Optional[str] = None,
        target_id: Optional[str] = None,
        target_curie: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Get the probability of having a relation between two entities."""
        source_id = self._lookup_node(node_id=source_id, node_curie=source_curie, node_name=source_name)
        target_id = self._lookup_node(node_id=target_id, node_curie=target_curie, node_name=target_name)
        lor = self.get_edge_probability(source_id, target_id)
        return {
            'source': self._get_entity_json(source_id),
            'target': self._get_entity_json(target_id),
            'lor': round(lor, self.precision),
        }

    def get_edge_embedding(self, source_id: str, target_id: str) -> np.ndarray:
        """Get the embedding of the edge between the two nodes."""
        return self.embeddings[source_id] * self.embeddings[target_id]

    def get_edge_probability(self, source_id: str, target_id: str) -> float:
        """Get the probability of the edge between the two nodes."""
        edge_embedding = self.get_edge_embedding(source_id, target_id)
        return self._predict_helper([edge_embedding.tolist()])[0]

    def _find_relations_helper(
        self,
        *,
        source_id: str,
        namespace: Optional[str] = None,
        source_vector: np.ndarray,
    ) -> RelationsResults:
        node_list, relations_list, relation_novelties = [], [], []
        for target_id, target_vector in self.embeddings.items():
            if source_id == target_id:
                continue

            novel = (
                self.graph is None or
                not (self.graph.has_edge(source_id, target_id) or self.graph.has_edge(target_id, source_id))
            )

            node_info = self._get_entity_json(target_id)
            if namespace is not None and node_info['namespace'] != namespace:
                continue
            node_list.append(node_info)
            relation_novelties.append(novel)

            # apply that hadamard operator
            relation = source_vector * target_vector
            relations_list.append(relation.tolist())

        return node_list, relations_list, relation_novelties

    def get_probabilities(
        self,
        *,
        nodes,
        relations: List[np.ndarray],
        relation_novelties: List[bool],
        k: Optional[int] = None,
    ) -> List[Mapping[str, Any]]:
        """Get probabilities from logistic regression classifier.

        Get the probabilities of all the relations in the list from the log model.
        Also sort the found probabilities by highest to lowest, then return the k highest probabilities.

        :param nodes: the list of the nodes with relations to the entity
        :param relations: the list of edge embedding of the two nodes
        :param k: the number of relations to be output
        :return: the k first probabilities in the list, type= list of tuples
        """
        probabilities = self._predict_helper(relations)
        results = [
            {
                'lor': round(lor, self.precision),
                'novel': novel,
                **node,
            }
            for node, lor, novel in zip(nodes, probabilities, relation_novelties)
        ]
        results = sorted(results, key=itemgetter('lor'))
        if not self.positive_control:
            # results = [result for result in results if result['novel']]
            results = list(filter(itemgetter('novel'), results))

        if k is not None:
            return results[:k]
        else:
            return results
