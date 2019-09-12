# -*- coding: utf-8 -*-

"""Create the flask application for :mod:`seffnet`."""

import logging
import os
from typing import Optional

from flasgger import Swagger
from flask import Flask
from flask_bootstrap import Bootstrap

from ..find_relations import Predictor

__all__ = [
    'create_app',
]

logger = logging.getLogger(__name__)


def create_app(predictor: Optional[Predictor] = None) -> Flask:
    """Make the :mod:`seffnet` web app."""
    app = Flask(__name__)
    app.secret_key = os.urandom(8)

    if predictor is None:
        from ..default_predictor import predictor
    app.config['predictor'] = predictor

    Swagger(app)
    Bootstrap(app)

    return app
