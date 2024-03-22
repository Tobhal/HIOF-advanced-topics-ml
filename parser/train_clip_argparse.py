import argparse
from os.path import join as ospj

def train_clip_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    train_clip_parser = parser.add_argument_group('Train clip arguments')
    
    # Model
    train_clip_parser.add_argument('-n', '--name', type=str, required=True)
    train_clip_parser.add_argument('--clip_model_name', type=str, default='ViT-B/32')

    # Config
    train_clip_parser.add_argument(
        '--config_dir', 
        type=str, 
        default=ospj('train_clip', 'models', 'configs', 'ViT.yaml')
    )

    # Optimizer
    train_clip_parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    train_clip_parser.add_argument('-wd', '--weight_decay', type=float, default=0.2)
    train_clip_parser.add_argument('-eps', '--epsilon', type=float, default=1e-8)

    # Scheduler
    train_clip_parser.add_argument('--cycle_mult', type=float, default=1.0)
    train_clip_parser.add_argument('--warmup_steps', type=int, default=2_000)

    # Data
    train_clip_parser.add_argument('--batch_size', type=int, default=32)

    # Training
    train_clip_parser.add_argument('--num_epochs', type=int, default=100)

    return parser

