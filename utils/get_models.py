from timm import create_model
import torch
from flags import device
from modules.utils import set_phos_version, set_phoc_version, gen_shape_description


def get_phoscnet(args) -> torch.nn.Module:
    """
    Get the phosc model.

    args:
        args: argparse.Namespace: The arguments to use for the model.

    Returns:
        torch.nn.Module: The phosc model.

    """
    phosc_model = create_model(
        model_name='resnet18phosc',
        phos_size=args.phos_size,
        phoc_size=args.phoc_size,
        phos_layers=args.phos_layers,
        phoc_layers=args.phoc_layers,
        dropout=args.dropout
    ).to(device)

    set_phos_version(args.phosc_version)
    set_phoc_version(args.phosc_version)

    return phosc_model
