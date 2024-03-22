import shutil
from os.path import join as ospj
import yaml
import sys
from os import PathLike


def load_args(filename: PathLike, args):
    """
    Load arguments from a yaml file and set them as attributes of the args object.

    Args:
        filename (str): The path to the yaml file.
        args (argparse.Namespace): The object to which the arguments should be set.
    """
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)
