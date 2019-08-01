# -*- coding: utf-8 -*-

"""Forms encoded in WTForms for ``se_kge``."""

from flask_wtf import FlaskForm
from wtforms.fields import RadioField, StringField, SubmitField
from wtforms.validators import DataRequired

__all__ = [
    'QueryForm',
]


class QueryForm(FlaskForm):
    """Builds the form for querying the model."""

    entity_identifier = StringField('Entity', validators=[DataRequired()])
    entity_type = RadioField(
        'Type',
        choices=[
            ('phenotype', 'Look for side effects'),
            ('target', 'Look for targets'),
        ],
        default='phenotype',
    )
    submit_subgraph = SubmitField('Submit')
