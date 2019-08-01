# -*- coding: utf-8 -*-

"""Create the API."""

from functools import lru_cache

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from .forms import QueryForm
from ..find_relations import find_new_relations

__all__ = [
    'api',
]

api = Blueprint('api', __name__)


@api.route('/', methods=['GET', 'POST'])
def home():
    """Show the home page."""
    form = QueryForm()

    if not form.validate_on_submit():
        test_url = url_for('.find', entity_identifier='85')
        return render_template('index.html', test_url=test_url, form=form)

    return redirect(url_for(
        '.find',
        entity_identifier=form.entity_identifier.data,
        entity_type=form.entity_type.data,
    ))


@lru_cache(maxsize=1000)
def find_relations_proxy(entity_identifier, entity_type, k):
    """Return memoized results for finding new relations."""
    return find_new_relations(
        entity_identifier=entity_identifier,
        embeddings=current_app.config['embeddings'],
        node_mapping=current_app.config['node_mapping'],
        saved_model=current_app.config['model'],
        graph=current_app.config['graph'],
        entity_type=entity_type,
        k=k,
    )


@api.route('/find/<entity_identifier>')
def find(entity_identifier):
    """Find new entities.

    ---
    parameters:
      - name: entity_identifier
        in: path
        description: The entity's CURIE
        required: true
        type: string
      - name: entity_type
        in: query
        description: The type of the entities for the incedent relations that get predicted
        required: false
        type: string
      - name: k
        in: query
        description: The number of predictions to return
        required: false
        type: integer

    """
    entity_type = request.args.get('entity_type', 'phenotype')
    k = request.args.get('k', 30, type=int)

    res = find_relations_proxy(
        entity_identifier=entity_identifier,
        entity_type=entity_type,
        k=k,
    )

    return jsonify(
        query=dict(
            entity_identifier=entity_identifier,
            entity_type=entity_type,
            k=k,
        ),
        result=res,
    )
