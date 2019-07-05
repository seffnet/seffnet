from bionev import pipeline
from bionev import embed_train
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import optuna

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file. Only accepted edgelist format.')
    parser.add_argument('--training', required=True,
                        help='training graph file. Only accepted edgelist format.')
    parser.add_argument('--testing', required=True,
                        help='testing graph file. Only accepted edgelist format.')
    args = parser.parse_args()

    return args

def node2vecOptimization(args):
    def objective(trial):
        dimensions = trial.suggest_int('dimensions', 100, 300)
        walk_length = trial.suggest_int('walk_length', 16, 128)
        number_walks = trial.suggest_int('number_walks', 16, 128)
        window_size = trial.suggest_int('window_size', 1, 10)
        p = trial.suggest_uniform('p', 0, 4.0)
        q = trial.suggest_uniform('p', 0, 4.0)
        G, G_train, testing_pos_edges, train_graph_filename = pipeline.train_test_graph(
            args.input,
            args.training,
            args.testing)
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
                                                                         test_pos_edges=testing_pos_edges, seed=0)
        trial.set_user_attr('mcc', mcc)
        trial.set_user_attr('auc_roc', auc_roc)
        trial.set_user_attr('auc_pr', auc_pr)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        return 1.0 - mcc

    study = optuna.create_study(storage='sqlite:///node2vec.db')
    study.set_user_attr('Author', 'Rana')
    study.set_user_attr('Method', 'node2vec')
    study.set_user_attr('Date', '05.07.2019')
    study.optimize(objective, n_trials=20)

if __name__ == "__main__":
    node2vecOptimization(parse_args())
