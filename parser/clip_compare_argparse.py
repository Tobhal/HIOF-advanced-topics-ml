import argparse
from os.path import join as ospj


def clip_compare_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments to the parser for the clip compare script.

    Args:
        parser (argparse.ArgumentParser): The parser to which the arguments should be added.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument('--clip_compare_config', type=str, default=ospj('configs', 'clip_compare', 'compare.yml'), help='Path to the clip compare configuration file')

    clip_compare_parser = parser.add_argument_group('Clip compare arguments')

    clip_compare_parser.add_argument('--name', type=str, default='clip-compare', help='name of the experiment')
    clip_compare_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    clip_compare_parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    return parser
