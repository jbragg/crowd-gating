import copy
import argparse
import json
import os
from . import constants
from . import Gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('history', type=argparse.FileType('r'), nargs='+')
    parser.add_argument(
        '--desired_accuracy', '-a', type=float,
        default=constants.DEFAULT_GATING_PARAMS['desired_accuracy'],
    )
    parser.add_argument(
        '--n_tutorial', type=int,
        default=constants.DEFAULT_GATING_PARAMS['n_tutorial'],
    )
    parser.add_argument(
        '--n_screening', type=int,
        default=constants.DEFAULT_GATING_PARAMS['n_screening'],
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=constants.DEFAULT_GATING_PARAMS['batch_size'],
    )
    parser.add_argument(
        '--gold_per_batch', type=int,
        default=constants.DEFAULT_GATING_PARAMS['gold_per_batch'],
    )
    parser.add_argument(
        '--n_gold_sliding', type=int,
        default=constants.DEFAULT_GATING_PARAMS['n_gold_sliding'],
    )
    parser.add_argument(
        '--no_exponential_backoff', action='store_true',
        help='Do not use exponential backoff in gating',
    )

    parser.add_argument(
        '--seed', '-s', type=int,
        help='Integer seed for randomization',
    )
    parser.add_argument(
        '--policy_path', '-p', type=str,
        help='Path to Guru policy',
    )
    parser.add_argument(
        '--policy_resolve', action='store_true',
        help='Write new policy to policy path. Defaults to reading only.',
    )
    args = parser.parse_args()

    if args.policy_resolve:
        raise NotImplementedError
    if len(args.history) > 1:
        raise NotImplementedError
    if args.policy_resolve and len(args.history) < 2:
        raise Exception("You probably don't want to resolve with so few histories.")
    if not args.policy_resolve and len(args.history) > 1:
        raise Exception('Action recommendations only work for a single history.')

    histories = [json.load(f) for f in args.history]

    gate = Gate(
        desired_accuracy=args.desired_accuracy,
        n_gold_sliding=args.n_gold_sliding,
        batch_size=args.batch_size,
        gold_per_batch=args.gold_per_batch,
        exponential_backoff=not args.no_exponential_backoff,
        n_tutorial=args.n_tutorial,
        n_screening=args.n_screening,
        seed=args.seed,
    )
    print(gate.next(histories[0]))
