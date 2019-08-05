# -*- coding: utf-8 -*-

"""Utilities for ``se_kge``."""

from typing import Any, Mapping

import optuna


def study_to_json(study: optuna.Study) -> Mapping[str, Any]:
    """Serialize a study to JSON."""
    return {
        'n_trials': len(study.trials),
        'name': study.study_name,
        'id': study.study_id,
        'start': study.start_datetime,
        'best': {
            'trial_attr': study.best_trial.user_attrs,
            'params': study.best_params,
            'trial': study.best_trial.number,
            'value': study.best_value,
        },
    }
