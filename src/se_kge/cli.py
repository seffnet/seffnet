from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from bionev import pipeline

from se_kge.optimization import hope_optimization, deepwalk_optimization, node2vec_optimization, grarep_optimization, \
    sdne_optimization, line_optimization


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file. Only accepted edgelist format.')
    parser.add_argument('--training', required=True,
                        help='training graph file. Only accepted edgelist format.')
    parser.add_argument('--testing', required=True,
                        help='testing graph file. Only accepted edgelist format.')
    parser.add_argument('--method', required=True,
                        help='The embedding learning method')
    parser.add_argument('--trials', default=50, type=int,
                        help='the number of trials done to optimize hyperparameters')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    G, G_train, testing_pos_edges, train_graph_filename = pipeline.train_test_graph(
        args.input,
        args.training,
        args.testing)
    if args.method == 'HOPE':
        hope_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials)

    elif args.method == 'DeepWalk':
        deepwalk_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials
        )

    elif args.method == 'node2vec':
        node2vec_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials)

    elif args.method == 'GraRep':
        grarep_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials
        )

    elif args.method == 'SDNE':
        sdne_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials
        )

    elif args.method == 'LINE':
        line_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials
        )



if __name__ == "__main__":
    main()
