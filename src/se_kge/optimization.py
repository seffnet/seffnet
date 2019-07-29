import datetime

import optuna
from bionev import embed_train, pipeline


def hope_optimization(*, G, G_train, testing_pos_edges, train_graph_filename, trial_number, seed):
    def objective(trial):
        dimensions = trial.suggest_int('dimensions', 100, 300)
        model = embed_train.train_embed_hope(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions)
        embeddings = model.get_embeddings()
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(embeddings=embeddings,
                                                                         original_graph=G,
                                                                         train_graph=G_train,
                                                                         test_pos_edges=testing_pos_edges, seed=seed)
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///hope.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'HOPE')
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=trial_number)


def deepwalk_optimization(*, G, G_train, testing_pos_edges, train_graph_filename, trial_number, seed):
    def objective(trial):
        dimensions = trial.suggest_int('dimensions', 100, 300)
        walk_length = trial.suggest_int('walk_length', 16, 128)
        number_walks = trial.suggest_int('number_walks', 16, 128)
        window_size = trial.suggest_int('window_size', 1, 10)
        model = embed_train.train_embed_deepwalk(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            walk_length=walk_length,
            number_walks=number_walks,
            window_size=window_size)
        embeddings = model.get_embeddings()
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(embeddings=embeddings,
                                                                         original_graph=G,
                                                                         train_graph=G_train,
                                                                         test_pos_edges=testing_pos_edges, seed=seed)
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///deepwalk.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'DeepWalk')
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=trial_number)


def node2vec_optimization(*, G, G_train, testing_pos_edges, train_graph_filename, trial_number, seed):
    def objective(trial):
        dimensions = trial.suggest_int('dimensions', 100, 300)
        walk_length = trial.suggest_int('walk_length', 16, 128)
        number_walks = trial.suggest_int('number_walks', 16, 128)
        window_size = trial.suggest_int('window_size', 1, 10)
        p = trial.suggest_uniform('p', 0, 4.0)
        q = trial.suggest_uniform('q', 0, 4.0)
        model = embed_train.train_embed_node2vec(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            walk_length=walk_length,
            number_walks=number_walks,
            window_size=window_size,
            p=p,
            q=q)
        embeddings = model.get_embeddings()
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(embeddings=embeddings,
                                                                         original_graph=G,
                                                                         train_graph=G_train,
                                                                         test_pos_edges=testing_pos_edges, seed=seed)
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///node2vec.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'node2vec')
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=trial_number)


def sdne_optimization(*, G, G_train, testing_pos_edges, train_graph_filename, trial_number, seed):
    def objective(trial):
        alpha = trial.suggest_uniform('alpha', 0, 0.4)
        beta = trial.suggest_int('beta', 0, 30)
        epochs = trial.suggest_int('epochs', 5, 30)
        model = embed_train.train_embed_sdne(
            train_graph_filename=train_graph_filename,
            alpha=alpha,
            beta=beta,
            epochs=epochs)
        embeddings = model.get_embeddings()
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(embeddings=embeddings,
                                                                         original_graph=G,
                                                                         train_graph=G_train,
                                                                         test_pos_edges=testing_pos_edges, seed=seed)
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///sdne.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'SDNE')
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=trial_number)


def grarep_optimization(*, G, G_train, testing_pos_edges, train_graph_filename, trial_number, seed):
    def objective(trial):
        dimensions = trial.suggest_int('dimensions', 100, 300)
        kstep = trial.suggest_int('kstep', 1, 10)
        model = embed_train.train_embed_grarep(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            kstep=kstep)
        embeddings = model.get_embeddings()
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(embeddings=embeddings,
                                                                         original_graph=G,
                                                                         train_graph=G_train,
                                                                         test_pos_edges=testing_pos_edges, seed=seed)
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///grarep.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'GraRep')
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=trial_number)


def line_optimization(*, G, G_train, testing_pos_edges, train_graph_filename, trial_number, seed):
    def objective(trial):
        dimensions = trial.suggest_int('dimensions', 100, 300)
        order = trial.suggest_int('order', 1, 3)
        epochs = trial.suggest_int('epochs', 5, 30)
        model = embed_train.train_embed_line(
            train_graph_filename=train_graph_filename,
            dimensions=dimensions,
            order=order,
            epochs=epochs)
        embeddings = model.get_embeddings_train()
        auc_roc, auc_pr, accuracy, f1, mcc = pipeline.do_link_prediction(embeddings=embeddings,
                                                                         original_graph=G,
                                                                         train_graph=G_train,
                                                                         test_pos_edges=testing_pos_edges, seed=seed)
        trial.set_user_attr('seed', seed)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///line.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'LINE')
    study.set_user_attr('Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    study.optimize(objective, n_trials=trial_number)
