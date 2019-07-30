# -*- coding: utf-8 -*-

"""Constants for ``se_kge``."""

import os

__all__ = [
    'HERE',
    'RESOURCES',
]

HERE = os.path.abspath(os.path.dirname(__file__))
RESOURCES = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'resources'))
