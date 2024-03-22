import argparse
from os.path import join as ospj


def clip_matrix_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments to the parser for the clip matrix script.

    Args:
        parser (argparse.ArgumentParser): The parser to which the arguments should be added.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument('--clip_matrix_config', type=str, default=ospj('configs', 'clip_matrix', 'compare.yml'), help='Path to the clip compare configuration file')

    clip_matrix_parser = parser.add_argument_group('Clip matrix arguments')

    clip_matrix_parser.add_argument('--eval_type', type=str, choices=['text', 'image'], default='text', help='Type of evaluation')

    return parser
