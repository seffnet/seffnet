# -*- coding: utf-8 -*-

"""Hyperparameter optimization for NRL models."""

import datetime
from getpass import getuser
from typing import Optional, Union

import optuna
from bionev import embed_train, pipeline
from bionev.OpenNE.line import LINE
from optuna import Study
from optuna.storages import BaseStorage


def run_study(
        objective,
        n_trials: int,
        *,
        storage: Union[None, str, BaseStorage] = None,
        study_name: Optional[str] = None,
) -> Study:
    """Run study for models.

    :param objective: the function for running the trial
    :param n_trials: the number of trials to run
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: returns the study
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize'
    )
    study.set_user_attr('Author', getuser())
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=n_trials)
    return study


def predict_and_evaluate(
        *,
        prediction_task,
        model,
        graph,
        graph_train,
        testing_pos_edges,
        seed,
        trial,
        labels,
        node_list,
        classifier_type: Optional[str] = None
):
    """Predict and evaluate the NRL model.

    :param model: NRL model
    :param graph: the complete graph
    :param graph_train: the graph to train the model
    :param testing_pos_edges: the edges used for testing
    :param seed: random seed
    :param trial: the trial used for the model
    :return: the value needed for minimization
    """
    if isinstance(model, LINE):
        embeddings = model.get_embeddings_train()
    else:
        embeddings = model.get_embeddings()
    if prediction_task == 'link_prediction':
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(
            embeddings=embeddings,
            original_graph=graph,
            train_graph=graph_train,
            test_pos_edges=testing_pos_edges,
            seed=seed,
            classifier_type=classifier_type
        )
        trial.set_user_attr('mcc', round(mcc, 3))
        trial.set_user_attr('auc_roc', round(auc_roc, 3))
        trial.set_user_attr('auc_pr', round(auc_pr, 3))
        trial.set_user_attr('accuracy', round(accuracy, 3))
        trial.set_user_attr('f1', round(f1, 3))
        return round(mcc, 3)
    else:
        accuracy, micro_f1, macro_f1 = pipeline.do_node_classification(
            embeddings=embeddings,
            node_list=node_list,
            labels=labels,
            seed=seed,
            classifier_type=classifier_type
        )
        trial.set_user_attr('accuracy', round(accuracy, 3))
        trial.set_user_attr('micro_f1', round(micro_f1, 3))
        trial.set_user_attr('macro_f1', round(macro_f1, 3))
        return round(micro_f1, 3)


def hope_optimization(
        *,
        graph,
        graph_train,
        testing_pos_edges,
        train_graph_filename,
        trial_number,
        seed,
        dimensions_range,
        storage=None,
        prediction_task,
        node_list,
        labels,
        classifier_type,
        study_name: Optional[str] = None,
        weighted: bool = False,
) -> Study:  # noqa: D202
    """Optimize HOPE method.

    :param graph: the complete input graph
    :param graph_train: the training graph
    :param testing_pos_edges:  the testing edges
    :param train_graph_filename: the path for training graph
    :param trial_number: the number of trials
    :param seed: the seed
    :param dimensions_range: the range for dimension parameter
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: the study
    """

    def objective(trial):
        trial.set_user_attr('method', 'hope')
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('classifier', classifier_type)

        dimensions = trial.suggest_int('dimensions', dimensions_range[0], dimensions_range[1])
        model = embed_train.train_embed_hope(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            weighted=weighted,
        )
        return predict_and_evaluate(
            model=model,
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            seed=seed,
            trial=trial,
            labels=labels,
            node_list=node_list,
            classifier_type=classifier_type,
            prediction_task=prediction_task,
        )

    return run_study(objective, trial_number, storage=storage, study_name=study_name)


def deepwalk_optimization(
        *,
        graph,
        graph_train,
        testing_pos_edges,
        train_graph_filename,
        trial_number,
        seed,
        dimensions_range,
        storage=None,
        prediction_task,
        node_list,
        labels,
        classifier_type,
        study_name: Optional[str] = None,
        weighted: bool = False,
) -> Study:  # noqa: D202
    """Optimize DeepWalk method.

    :param graph: the complete input graph
    :param graph_train: the training graph
    :param testing_pos_edges:  the testing edges
    :param train_graph_filename: the path for training graph
    :param trial_number: the number of trials
    :param seed: the seed
    :param dimensions_range: the range for dimension parameter
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: the study
    """

    def objective(trial):
        trial.set_user_attr('method', 'deepwalk')
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('classifier', classifier_type)

        dimensions = trial.suggest_int('dimensions', dimensions_range[0], dimensions_range[1])
        walk_length = trial.suggest_categorical('walk_length', [8, 16, 32, 64, 128])
        number_walks = trial.suggest_categorical('number_walks', [8, 16, 32, 64, 128, 256])
        window_size = trial.suggest_int('window_size', 2, 6)
        model = embed_train.train_embed_deepwalk(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            walk_length=walk_length,
            number_walks=number_walks,
            window_size=window_size,
            weighted=weighted,
        )
        return predict_and_evaluate(
            prediction_task=prediction_task,
            model=model,
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            seed=seed,
            trial=trial,
            labels=labels,
            node_list=node_list,
            classifier_type=classifier_type,
        )

    return run_study(objective, trial_number, storage=storage, study_name=study_name)


def node2vec_optimization(
        *,
        graph,
        graph_train,
        testing_pos_edges,
        train_graph_filename,
        trial_number,
        seed,
        dimensions_range,
        storage=None,
        prediction_task,
        node_list,
        labels,
        classifier_type,
        weighted=False,
        study_name: Optional[str] = None,
) -> Study:  # noqa: D202
    """Optimize node2vec method.

    :param graph: the complete input graph
    :param graph_train: the training graph
    :param testing_pos_edges:  the testing edges
    :param train_graph_filename: the path for training graph
    :param trial_number: the number of trials
    :param seed: the seed
    :param dimensions_range: the range for dimension parameter
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: the study
    """

    def objective(trial):
        trial.set_user_attr('method', 'node2vec')
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('classifier', classifier_type)

        dimensions = trial.suggest_int('dimensions', dimensions_range[0], dimensions_range[1])
        walk_length = trial.suggest_categorical('walk_length', [8, 16, 32, 64, 128])
        number_walks = trial.suggest_categorical('number_walks', [8, 16, 32, 64, 128, 256])
        window_size = trial.suggest_int('window_size', 2, 6)
        p = trial.suggest_uniform('p', 0, 4.0)
        q = trial.suggest_uniform('q', 0, 4.0)
        model = embed_train.train_embed_node2vec(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            walk_length=walk_length,
            number_walks=number_walks,
            window_size=window_size,
            p=p,
            q=q,
            weighted=weighted,
        )
        return predict_and_evaluate(
            prediction_task=prediction_task,
            model=model,
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            seed=seed,
            trial=trial,
            labels=labels,
            node_list=node_list,
            classifier_type=classifier_type,
        )

    return run_study(objective, trial_number, storage=storage, study_name=study_name)


def sdne_optimization(
        *,
        graph,
        graph_train,
        testing_pos_edges,
        train_graph_filename,
        trial_number,
        seed,
        storage=None,
        prediction_task,
        node_list,
        labels,
        classifier_type,
        weighted=False,
        study_name: Optional[str] = None,
) -> Study:  # noqa: D202
    """Optimize SDNE method.

    :param graph: the complete input graph
    :param graph_train: the training graph
    :param testing_pos_edges:  the testing edges
    :param train_graph_filename: the path for training graph
    :param trial_number: the number of trials
    :param seed: the seed
    :param dimensions_range: the range for dimension parameter
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: the study
    """

    def objective(trial):
        trial.set_user_attr('method', 'sdne')
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('classifier', classifier_type)

        alpha = trial.suggest_uniform('alpha', 0, 0.4)
        beta = trial.suggest_int('beta', 0, 30)
        epochs = trial.suggest_categorical('epochs', [5, 10, 15, 20, 25, 30])
        model = embed_train.train_embed_sdne(
            train_graph_filename=train_graph_filename,
            alpha=alpha,
            beta=beta,
            epochs=epochs,
            weighted=weighted,
        )
        return predict_and_evaluate(
            prediction_task=prediction_task,
            model=model,
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            seed=seed,
            trial=trial,
            labels=labels,
            node_list=node_list,
            classifier_type=classifier_type,
        )

    return run_study(objective, trial_number, storage=storage, study_name=study_name)


def grarep_optimization(
        *,
        graph,
        graph_train,
        testing_pos_edges,
        train_graph_filename,
        trial_number,
        seed,
        dimensions_range,
        storage=None,
        prediction_task,
        node_list,
        labels,
        classifier_type,
        weighted=False,
        study_name: Optional[str] = None,
) -> Study:  # noqa: D202
    """Optimize GraRep method.

    :param graph: the complete input graph
    :param graph_train: the training graph
    :param testing_pos_edges:  the testing edges
    :param train_graph_filename: the path for training graph
    :param trial_number: the number of trials
    :param seed: the seed
    :param dimensions_range: the range for dimension parameter
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: the study
    """

    def objective(trial):
        trial.set_user_attr('method', 'grarep')
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('classifier', classifier_type)
        # TODO: need to choose kstep in which it can divide the dimension
        dimensions = trial.suggest_int('dimensions', dimensions_range[0], dimensions_range[1])
        kstep = trial.suggest_int('kstep', 1, 7)
        if dimensions % kstep != 0:
            raise optuna.structs.TrialPruned()
        model = embed_train.train_embed_grarep(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            kstep=kstep,
            weighted=weighted,
        )
        return predict_and_evaluate(
            prediction_task=prediction_task,
            model=model,
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            seed=seed,
            trial=trial,
            labels=labels,
            node_list=node_list,
            classifier_type=classifier_type,
        )

    return run_study(objective, trial_number, storage=storage, study_name=study_name)


def line_optimization(
        *,
        graph,
        graph_train,
        testing_pos_edges,
        train_graph_filename,
        trial_number,
        seed,
        dimensions_range,
        storage=None,
        prediction_task,
        node_list,
        labels,
        classifier_type,
        weighted=False,
        study_name: Optional[str] = None,
) -> Study:  # noqa: D202
    """Optimize LINE method.

    :param graph: the complete input graph
    :param graph_train: the training graph
    :param testing_pos_edges:  the testing edges
    :param train_graph_filename: the path for training graph
    :param trial_number: the number of trials
    :param seed: the seed
    :param dimensions_range: the range for dimension parameter
    :param storage: the database that stores the studies and trials
    :param study_name: the name of the study
    :return: the study
    """

    def objective(trial):
        trial.set_user_attr('method', 'line')
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('classifier', classifier_type)

        dimensions = trial.suggest_int('dimensions', dimensions_range[0], dimensions_range[1])
        order = trial.suggest_categorical('order', [1, 2, 3])
        epochs = trial.suggest_categorical('epochs', [5, 10, 15, 20, 25, 30])
        model = embed_train.train_embed_line(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            order=order,
            epochs=epochs,
            weighted=weighted,
        )
        return predict_and_evaluate(
            prediction_task=prediction_task,
            model=model,
            graph=graph,
            graph_train=graph_train,
            testing_pos_edges=testing_pos_edges,
            seed=seed,
            trial=trial,
            labels=labels,
            node_list=node_list,
            classifier_type=classifier_type,
        )

    return run_study(objective, trial_number, storage=storage, study_name=study_name)
