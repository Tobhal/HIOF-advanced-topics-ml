import argparse
from flags import DATA_FOLDER
from os.path import join as ospj


def dataset_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments to the parser for the dataset script.

    Args:
        parser (argparse.ArgumentParser): The parser to which the arguments should be added.

    Returns:
        argparse.ArgumentParser: The parser with the added arguments.
    """
    parser.add_argument('--data_config', type=str, default=ospj('configs', 'data', 'default.yaml'), help='Path to the data configuration file')

    dataset_parser = parser.add_argument_group('Dataset arguments')

    dataset_parser.add_argument('--data_dir', default='mit-states', help='local path to data root dir from ' + DATA_FOLDER)
    dataset_parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos|bengali')
    dataset_parser.add_argument('--split_name', default='compositional-split-natural', help="dataset split")
    dataset_parser.add_argument('--image_extractor', default = 'resnet18', help = 'Feature extractor model')
    dataset_parser.add_argument('--num_negs', type=int, default=1, help='Number of negatives to sample per positive (triplet loss)')
    dataset_parser.add_argument('--pair_dropout', type=float, default=0.0, help='Each epoch drop this fraction of train pairs')
    dataset_parser.add_argument('--test_set', default='val', help='val|test mode')

    dataset_parser.add_argument('--subset', action='store_true', default=False, help='test on a 1000 image subset (debug purpose)')
    dataset_parser.add_argument('--train_only', action='store_true', help='Train only')
    dataset_parser.add_argument('--update_features', action='store_true', default=False, help='Update features')
    dataset_parser.add_argument('--return_images', action='store_true', default=False, help='Return images at the end of the data')
    dataset_parser.add_argument('--augmented', action='store_true', default=False, help='Augment data')
    dataset_parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')
    dataset_parser.add_argument('--add_original_data', action='store_true', default=False, help='Add original data to the dataset')

    dataloader_parser = parser.add_argument_group('Dataloader arguments')
    dataloader_parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    dataloader_parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    dataloader_parser.add_argument('--shuffled', action='store_true', default=True, help='shuffle the data')

    optimizer_parser = parser.add_argument_group('Optimizer arguments')
    optimizer_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    optimizer_parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
    optimizer_parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adam optimizer')

    learning_rate_parser = parser.add_argument_group('Learning rate scheduler arguments')
    learning_rate_parser.add_argument('--cycle_mult', type=float, default=1.0, help='cycle multiplier')

    """
    dataset_parser.add_argument('--config', default='configs/args.yml', help='path of the config file (training only)')
    dataset_parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
    dataset_parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
    dataset_parser.add_argument('--exp_name', default='temp', help='Name of exp used to name models')
    dataset_parser.add_argument('--load', default=None, help='path to checkpoint to load from')
    dataset_parser.add_argument('--norm_family', default = 'imagenet', help = 'Normalization values from dataset')
    dataset_parser.add_argument('--clean_only', action='store_true', default=False, help='use only clean subset of data (mitstates)')
    dataset_parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
    dataset_parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')
    """

    return parser
