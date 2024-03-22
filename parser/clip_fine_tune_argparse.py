import argparse
from os.path import join as ospj


def clip_fine_tune_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments to the parser for the clip fine-tune script.

    Args:
        parser (argparse.ArgumentParser): The parser to which the arguments should be added.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument('--clip_fine_tune_config', type=str, default=ospj('configs', 'clip', 'fine-tune.yml'), help='Path to the clip fine-tune configuration file')

    clip_fine_tune_parser = parser.add_argument_group('Align fine-tune arguments')

    clip_fine_tune_parser.add_argument('--name', type=str, default='align-fine-tune', help='name of the experiment')
    clip_fine_tune_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    clip_fine_tune_parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    return parser
