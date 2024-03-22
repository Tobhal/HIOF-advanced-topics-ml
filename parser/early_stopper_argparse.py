import argparse
from os.path import join as ospj


def early_stopper_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments to the parser for the early stopper script.

    Args:
        parser (argparse.ArgumentParser): The parser to which the arguments should be added.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument('--early_stopper_config', type=str, default=ospj('configs', 'early_stopper', 'default.yaml'), help='Path to the early stopper configuration file')

    early_stopper_parser = parser.add_argument_group('early stopper arguments')

    early_stopper_parser.add_argument('--stop_patience', type=int, default=7, help='patience for early stopping')
    early_stopper_parser.add_argument('--verbose', action='store_true', default=False, help='verbose for early stopping')
    early_stopper_parser.add_argument('--save_every', type=int, default=5, help='save model every n epochs')

    return parser
