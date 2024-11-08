"""Functions to launch training from the command line"""

import sys
import argparse
from typing import Optional
import logging

from src.libs.preprocessing import load_data

from src.model.experiments import init_model_from_config


def get_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "-e", "--exp", nargs="+", type=int, required=True, help="Experiment id"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--local", action="store_true", help="Load data from local filesystem"
    )
    return parser


def train_main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    df_learning = load_data(is_train=True, is_local=args.local)
    df_testing = load_data(is_train=False, is_local=args.local)
    for exp in args.exp:
        model = init_model_from_config(exp)
        logging.info(f"Training experiment { exp }")
        model.full_pipeline(df_learning=df_learning, df_testing=df_testing)


if __name__ == "__main__":
    train_main(sys.argv[1:])
