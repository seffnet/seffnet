from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import random

from bionev import pipeline
from .optimization import (
    deepwalk_optimization, grarep_optimization, hope_optimization, line_optimization,
    node2vec_optimization, sdne_optimization,
)


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
    parser.add_argument('--dimensions', nargs='+', default=[100, 300], type=int,
                        help='the range of dimensions to be optimized')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    G, G_train, testing_pos_edges, train_graph_filename = pipeline.train_test_graph(
        args.input,
        args.training,
        args.testing)
    seed = random.randint(1, 10000000)
    if args.method == 'HOPE':
        hope_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials,
            seed=seed,
            dimensions_range=args.dimensions
        )

    elif args.method == 'DeepWalk':
        deepwalk_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials,
            seed=seed,
            dimensions_range=args.dimensions

        )

    elif args.method == 'node2vec':
        node2vec_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials,
            seed=seed,
            dimensions_range=args.dimensions
        )

    elif args.method == 'GraRep':
        grarep_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials,
            seed=seed,
            dimensions_range=args.dimensions
        )

    elif args.method == 'SDNE':
        sdne_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials,
            seed=seed
        )

    elif args.method == 'LINE':
        line_optimization(
            G=G,
            G_train=G_train,
            testing_pos_edges=testing_pos_edges,
            train_graph_filename=train_graph_filename,
            trial_number=args.trials,
            seed=seed,
            dimensions_range=args.dimensions
        )


if __name__ == "__main__":
    main()
