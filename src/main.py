from pipelines import eval as eval_
from pipelines import train
from utils.config_loader import load_config

import argparse


def main(args):
    config = load_config(args.config)

    if args.pipeline == 'train':
        train.train_wrapper(config)
    elif args.pipeline == 'eval':
        eval_.eval_wrapper(config)

    else:
        raise ValueError(f'Unsupported pipeline: {args.pipeline}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Customizable Image Retrieval Framework'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to config file'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        required=True,
        help='Pipeline to run: train, eval',
    )
    args = parser.parse_args()
    main(args)
