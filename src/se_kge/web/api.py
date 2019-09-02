# -*- coding: utf-8 -*-

"""Create the API."""

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for
from werkzeug.local import LocalProxy

from .forms import QueryForm

__all__ = [
    'api',
]

api = Blueprint('api', __name__)

predictor = LocalProxy(lambda: current_app.config['predictor'])


def get_result(curie: str):
    """Get the prediction results using the current request."""
    results_type = request.args.get('results_type', 'phenotype')
    k = request.args.get('k', 30, type=int) or None

    return predictor.find_new_relations(
        node_curie=curie,
        results_type=results_type,
        k=k,
    )


@api.route('/', methods=['GET', 'POST'])
def home():
    """Show the home page."""
    form = QueryForm()

    if not form.validate_on_submit():
        test_url = url_for('.find', curie='pubchem:85')
        return render_template('index.html', test_url=test_url, form=form)

    return redirect(url_for(
        '.predict',
        curie=form.curie.data,
        results_type=form.results_type.data,
    ))


@api.route('/list')
def list_nodes():
    """Return all entities as JSON."""
    offset = request.args.get('offset', 0, type=int)
    pagesize = request.args.get('size', 30, type=int)

    v = list(predictor.node_id_to_info.values())
    nodes = v[offset:offset + pagesize]

    next_offset = offset + pagesize
    if next_offset > len(v):
        pagesize -= (next_offset - len(v))

    return jsonify(
        links={
            'next': url_for('.list_nodes', offset=next_offset, pagesize=pagesize),
            'last': url_for('.list_nodes', offset=offset - pagesize, pagesize=pagesize),
        },
        nodes=nodes,
    )


@api.route('/predict/<curie>')
def predict(curie: str):
    """Predict edges for the given entity.

    ---
    parameters:
      - name: node_curie
        in: path
        description: The entity's CURIE
        required: true
        type: string
      - name: results_type
        in: query
        description: The type of the entities for the incident relations that get predicted
        required: false
        type: string
      - name: k
        in: query
        description: The number of predictions to return
        required: false
        type: integer
      - name: format
        in: query
        description: The type of result to return. If json, return results as JSON.
        required: false
        type: string

    """
    result = get_result(curie)
    return_format = request.args.get('format', 'html')

    if return_format is None or return_format == 'html':
        return render_template(
            'predictions.html',
            curie=curie,
            results_type=result['query']['type'],
            k=result['query']['k'],
            predictions=result['predictions'],
        )

    elif return_format == 'json':
        if result is None:
            return jsonify({
                'query': {
                    'curie': curie,
                    'results_type': result['query']['type'],
                    'k': result['query']['k'],
                },
                'message': 'Not found',
            })

        # Add those sweet sweet identifiers.org links
        for prediction in result['predictions']:
            prediction['url'] = f'https://identifiers.org/{prediction["namespace"]}:{prediction["identifier"]}'

        return jsonify(result)

    else:
        return abort(500, f'Invalid return type: {return_format}')
