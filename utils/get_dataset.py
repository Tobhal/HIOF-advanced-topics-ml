from torch.utils.data import DataLoader
from data import dataset_bengali as dset
from flags import DATA_FOLDER
from os.path import join as ospj

def get_training_loader(args, phosc_net_model=None) -> DataLoader:
    # Get dataset
    train_set = dset.CompositionDataset(
        root=ospj(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.split_name,
        model=args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only=args.train_only,
        open_world=args.open_world,
        add_original_data=True,
        augmented=args.augmented,
        phosc_model=phosc_net_model,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=args.shuffled,
        num_workers=args.workers
    )

    return train_loader, train_set


def get_validation_loader(args, phosc_net_model=None) -> DataLoader:
    # Get dataset
    val_set = dset.CompositionDataset(
        root=ospj(DATA_FOLDER,args.data_dir),
        phase='val',
        split=args.split_name,
        model=args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only=args.train_only,
        open_world=args.open_world,
        add_original_data=True,
        augmented=False,
        phosc_model=phosc_net_model,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=args.shuffled,
        num_workers=args.workers
    )

    return val_loader, val_set


def get_test_loader(args, phosc_net_model=None) -> DataLoader:
    # Get dataset
    test_set = dset.CompositionDataset(
        root=ospj(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.split_name,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world,
        add_original_data=True,
        augmented=False,
        phosc_model=phosc_net_model,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=args.shuffled,
        num_workers=args.workers
    )

    return test_loader, test_set