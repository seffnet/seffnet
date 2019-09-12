# -*- coding: utf-8 -*-

"""Default predictor for :mod:`seffnet`."""

from .constants import DEFAULT_EMBEDDINGS_PATH, DEFAULT_GRAPH_PATH, DEFAULT_MAPPING_PATH, DEFAULT_MODEL_PATH
from .find_relations import Predictor

__all__ = [
    'predictor',
]

predictor = Predictor.from_paths(
    model_path=DEFAULT_MODEL_PATH,
    embeddings_path=DEFAULT_EMBEDDINGS_PATH,
    graph_path=DEFAULT_GRAPH_PATH,
    mapping_path=DEFAULT_MAPPING_PATH,
    positive_control=False,
)
