# -*- coding: utf-8 -*-

"""Create the API."""

from functools import lru_cache
from typing import Optional

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from .forms import QueryForm
from ..find_relations import Predictor

__all__ = [
    'api',
]

api = Blueprint('api', __name__)


@api.route('/', methods=['GET', 'POST'])
def home():
    """Show the home page."""
    form = QueryForm()

    if not form.validate_on_submit():
        test_url = url_for('.find', curie='pubchem:85')
        return render_template('index.html', test_url=test_url, form=form)

    return redirect(url_for(
        '.find',
        entity_identifier=form.entity_identifier.data,
        entity_type=form.entity_type.data,
    ))


@lru_cache(maxsize=1000)
def find_relations_proxy(node_curie: str, result_type: Optional[str] = None, k: Optional[int] = 30):
    """Return memoized results for finding new relations."""
    predictor: Predictor = current_app.config['predictor']
    return predictor.find_new_relations(
        node_curie=node_curie,
        result_type=result_type,
        k=k,
    )


@api.route('/list')
def list_nodes():
    """Return all entities as JSON."""
    return jsonify(current_app.config['predictor'].node_id_to_info)


@api.route('/find/<curie>')
def find(curie: str):
    """Find new entities.

    ---
    parameters:
      - name: node_curie
        in: path
        description: The entity's CURIE
        required: true
        type: string
      - name: entity_type
        in: query
        description: The type of the entities for the incident relations that get predicted
        required: false
        type: string
      - name: k
        in: query
        description: The number of predictions to return
        required: false
        type: integer

    """
    entity_type = request.args.get('result_type', 'phenotype')
    k = request.args.get('k', 30, type=int) or None

    result = find_relations_proxy(
        node_curie=curie,
        result_type=entity_type,
        k=k,
    )

    return jsonify(result)
