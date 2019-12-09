# -*- coding: utf-8 -*-

"""Query scientific literature through Europe PMC to identify co-occurrences (e.g., chemical + phenotype) in articles.

How to Use
----------

1. Using the cli for a single entity:

.. code-block:: sh

    python -m seffnet.literature Ropinirole umls:C0015371

2. Using the cli for combinations of entities:

.. code-block:: sh

    python -m seffnet.literature Ropinirole umls:C0015371 umls:C0013384

3. Using the main method in Python:

.. code-block:: python

    query_europe_pmc(
        query_entity='Ropinirole',
        target_entities=[
            'umls:C0013384',
            'umls:C0015371',
        ],
    )

Examples of namespaces and entities (more info at https://europepmc.org/AnnotationsAPI):

- obo/CHEBI_59905
- taxonomy/9103
- go/GO:0030431
- umls-concept/C0393777
- uniprot/P10995
- resource/SIO_000419

Examples of types:

- Gene Ontology
- Diseases
- Gene_Proteins
- Chemicals
"""

import logging
import os
import time
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import click
import requests

logging.basicConfig(level=logging.INFO)

API_EBI_ANNOTATIONS = 'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByEntity'

#: Key is the real namespace, value is the one expected by Europe PMC
_NAMESPACE_FIXER = {
    'umls': 'umls-concept',
}
_NAMESPACE_FIXER_INVERSE = {
    'umls-concept': 'umls',
}


@click.command()
@click.argument('query')
@click.argument('targets', nargs=-1)
def query(query, targets):
    """Query Europe PMC to identify co-occurrences in the literature."""
    targets = list(targets)
    click.echo(f'Querying articles containing "{query}" with the following co-occurrences: "{targets}""')

    results = query_europe_pmc(query, targets)

    articles_list = []

    for result in results:
        click.echo(result)
        articles_list.append(result)

    import json

    with open('result.json', 'w+') as fp:
        json.dump(
            {
                "query": query,
                "targets": targets,
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "articles": articles_list,
            }, fp,
        )
    click.echo(f"Results exported to {os.path.join(os.path.dirname(__file__), 'result.json')}")


def query_europe_pmc(
    query_entity: str,
    target_entities: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]],
) -> Iterable[Mapping[str, str]]:
    """Query Europe PMC API by entity.

    :param query_entity: query_entity string
    :param target_entities: A tuple or list of tuples describing the namespace/identifiers of the entities
    :return: list of articles matching the co-occurrence
    """
    if isinstance(target_entities, (tuple, str)):
        target_entities = [target_entities]

    target_entities = _clean_entity_tuples(target_entities)

    response_dict = _query_api(query_entity)
    yield from _get_matching_annotations_for_articles(
        query=query_entity,
        articles=response_dict['articles'],
        entity_tuples=target_entities,
    )

    cursor_mark = response_dict.get('nextCursorMark')
    while cursor_mark is not None and cursor_mark != -1.0:
        response_dict = _query_api(query_entity, cursor_mark)
        yield from _get_matching_annotations_for_articles(
            query=query_entity,
            articles=response_dict['articles'],
            entity_tuples=target_entities,
        )
        cursor_mark = response_dict.get('nextCursorMark')


def _clean_entity_tuples(tuples):
    x = [
        t.split(':') if isinstance(t, str) else t
        for t in tuples
    ]

    return [
        (_NAMESPACE_FIXER.get(namespace, namespace), identifier)
        for namespace, identifier in x
    ]


def _query_api(string: str, cursor_mark: Optional[str] = None, page_size: int = 8):
    """Query Europe PMC API.

    :param string: query string
    :param cursor_mark: optional parameter for pagination in Solr
    :return: JSON response as a dictionary
    """
    # First query
    params = {
        'entity': string,
        'filter': 0,
        'format': 'JSON',
        'pageSize': page_size,
    }

    # Add extra parameter for pagination
    if cursor_mark:
        params['cursorMark'] = cursor_mark

    # Query API
    response = requests.get(API_EBI_ANNOTATIONS, params=params)

    if response.status_code != 200:
        raise ConnectionError(
            f'Unsuccessful response {response.status_code} from EBI. '
            f'Please check that the EBI API works and that you have access to the Internet.',
        )

    return response.json()


def _get_matching_annotations_for_articles(
    *,
    query: str,
    articles: List[Dict],
    entity_tuples: List[Tuple[str, str]],
) -> Iterable[Mapping[str, str]]:
    for article in articles:
        matching_annotations = _get_matching_annotations_for_article(
            string=query,
            article=article,
            entity_tuples=entity_tuples,
        )

        # Article does not contain the annotation for 2nd entity
        if matching_annotations:
            yield matching_annotations


def _get_matching_annotations_for_article(
    *,
    string,
    article,
    entity_tuples: List[Tuple[str, str]],
) -> Optional[Mapping[str, str]]:
    matching_annotations = _check_annotation_present(
        article['annotations'],
        entity_tuples,
    )
    if matching_annotations:
        return {
            'searched term': string,
            'co-occurrence(s) found': [
                {
                    "namespace": _NAMESPACE_FIXER_INVERSE.get(namespace, namespace),
                    "identifier": identifier,
                    "exact_term": exact_term,
                }
                for namespace, identifier, exact_term in matching_annotations
            ],
            'source': article['source'],
            'extId': article['extId'],
            'pmcid': article.get('pmcid', 'NA'),
        }


def _check_annotation_present(
    annotations: Iterable[Dict],
    entity_tuples: List[Tuple[str, str]],
) -> Set[Tuple[str, str, str]]:
    """Check if annotation is present in the article.

    :param annotations: query string
    :param entity_tuples: list of tuples with namespace/identifiers
    :return: list of valid annotations in the article
    """
    valid_annotations = set()
    for annotation in annotations:

        # Iterate through the different tags of the entity
        for tags in annotation["tags"]:

            # If the entity matches one of the
            for namespace, identifier in entity_tuples:
                # Check that the URI matches the identifier and namespace
                if not tags["uri"].endswith(f'{namespace}/{identifier}'):
                    continue

                valid_annotations.add((namespace, identifier, annotation['exact']))

    return valid_annotations


if __name__ == '__main__':
    query()
