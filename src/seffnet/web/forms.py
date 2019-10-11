# -*- coding: utf-8 -*-

"""Forms encoded in WTForms."""

from flask_wtf import FlaskForm
from wtforms.fields import RadioField, StringField, SubmitField
from wtforms.validators import DataRequired

__all__ = [
    'QueryForm',
]


class QueryForm(FlaskForm):
    """Builds the form for querying the model."""

    curie = StringField('Entity', validators=[DataRequired()])
    results_type = RadioField(
        'Type',
        choices=[
            ('chemical', 'Look for chemicals'),
            ('phenotype', 'Look for side effects'),
            ('target', 'Look for targets'),
            ('everything', 'Look for everything'),
        ],
        default='phenotype',
    )
    submit_subgraph = SubmitField('Submit')
