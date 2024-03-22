import yaml
import copy
import math

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from flags import device, DATA_FOLDER

from os.path import join as ospj

from clip.clip import tokenize

# PHOSC utils
from modules.utils import set_phos_version, set_phoc_version
from modules.utils.utils import get_phosc_description
from utils.utils import text_features_from_description
from modules import models, residualmodels
from timm import create_model
from modules.utils import gen_shape_description_simple, gen_shape_description
from modules.utils.utils import split_string_into_chunks

import clip

from data import dataset_bengali as dset
from parser import train_clip_argparse, phosc_net_argparse, dataset_argparse, early_stopper_argparse
from utils.utils import load_args
from utils.dbe import dbe
from utils.early_stopping import EarlyStopping

from train_clip.utils.clip_utils import gen_word_objs_embeddings_batch

from data.dataset_bengali import ImageLoader

# CLIP
from train_clip.models.model import CLIP

from typing import List, Tuple

def num_training_steps(train_dataloader, max_epochs, batch_size, accumulate_grad_batches=1):
    dataset_size = len(train_dataloader.dataset)  # Assuming your dataloader has a dataset attribute

    effective_batch_size = batch_size * accumulate_grad_batches
    total_steps = (dataset_size // effective_batch_size) * max_epochs

    return total_steps


def train_step_batch(images, words, batch_size, clip_model, optimizer, lr_scheduler):
    # Transform setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Process images
    images = torch.stack([transform(img) for img in images])
    images = clip_model.encode_image(images)

    # Process words
    texts = text_features_from_description(words, clip_model)

    # Normalize image embeddings after encoding
    ims = [F.normalize(image, dim=0) for image in images]
    ims = torch.stack(ims)

    # Normalize text embeddings after encoding
    txt = [F.normalize(txt, dim=0) for txt in texts]
    txt = torch.stack(txt)

    # Compute similarity scores between images and texts
    image_logits = ims @ txt.t() * clip_model.logit_scale.exp()
    ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)

    # Compute loss
    total_loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)

    # Compute accuracy
    acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
    acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
    acc = (acc_i + acc_t).float() / 2 / len(images)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    # Clamp the logit scale
    clip_model.logit_scale.data = clip_model.logit_scale.data.clamp(-np.log(100), np.log(100))

    return total_loss.item(), acc.item()


def train_one_epoch(epoch: int, train_loader, clip_model, image_loader, optimizer, lr_scheduler) -> Tuple[float, float]:
    clip_model.train()
    running_loss = 0
    running_acc = 0

    train_bar = tqdm(train_loader, desc=f'Training epoch {epoch + 1}')
    for batch in train_bar:
        optimizer.zero_grad()

        *_, image_names, _, words = batch

        # words = [get_phosc_description(word) for word in words]
        images = [image_loader(image) for image in tqdm(image_names, position=1, desc='Processing Images', leave=False)]

        temp_loss, temp_acc = train_step_batch(images, words, 32, clip_model, optimizer, lr_scheduler)

        running_loss += temp_loss
        running_acc += temp_acc

    loss = running_loss / len(train_loader)
    acc = running_acc / len(train_loader)
    # loss.backward()

    optimizer.step()
    lr_scheduler.step()

    return loss, acc


def validation_step_batch(image, text, clip_model, batch_size: int) -> Tuple[float, float]:
    loss = 0
    acc = 0

    n = math.ceil(len(image) // batch_size)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = torch.stack([transform(img) for img in image])

    # Tokenize each text sample
    text_tokenized = [tokenize(txt).squeeze(0) for txt in text]
    text_tensor = torch.stack(text_tokenized)

    n = n if n > 0 else 1

    image_emb = torch.chunk(image, n)
    text_emb = torch.chunk(text_tensor, n)

    with torch.no_grad():
        ims = [F.normalize(clip_model.encode_image(img), dim=1) for img in image_emb]
        txt = [F.normalize(clip_model.encode_text(t), dim=1) for t in text_emb]

        ims = torch.cat(ims)
        txt = torch.cat(txt)

        if len(ims.shape) == 3:
            ims = list(ims)
            txt = list(txt)
        else:
            ims = [ims]
            txt = [txt]

        image_logits = torch.cat(ims) @ torch.cat(txt).t() * clip_model.logit_scale.exp()
        ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)

        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)

        acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
        acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

        loss = loss / len(ims)
        acc = (acc_i + acc_t) / 2 / len(image) / len(ims)

    return loss, acc


def validation_one_epoch(epoch: int, val_loader, clip_model, image_loader) -> Tuple[float, float]:
    clip_model.eval()
    running_loss = 0
    running_acc = 0

    val_bar = tqdm(val_loader, desc=f'Validation epoch {epoch + 1}')
    for batch in val_bar:
        *_, image_names, _, descriptions = batch

        images = [image_loader(image) for image in tqdm(image_names, position=1, desc='Processing Images', leave=False)]
        descriptions = [description for description in descriptions]

        temp_loss, temp_acc = validation_step_batch(images, descriptions, clip_model, len(batch))

        running_loss += temp_loss.item()
        running_acc += temp_acc.item()

    avg_loss = running_loss / len(val_loader)
    avg_acc = running_acc / len(val_loader)

    return avg_loss, avg_acc


def main():
    # Setup arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_clip_config', type=str, default=ospj('configs', 'train_clip', 'default.yaml'), help='Path to the training configuration file')
    parser.add_argument('--phosc_config', type=str, default=ospj('configs', 'phoscnet', 'default.yaml'), help='Path to the phoscnet configuration file')
    parser.add_argument('--data_config', type=str, default=ospj('configs', 'data', 'default.yaml'), help='Path to the data configuration file')
    parser.add_argument('--early_stopper_config', type=str, default=ospj('configs', 'early_stopper', 'default.yaml'), help='Path to the early stopper configuration file')

    parser = train_clip_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)

    args = parser.parse_args()

    # Load argumetns form file
    load_args(args.train_clip_config, args)
    load_args(args.phosc_config, args)
    load_args(args.data_config, args)
    load_args(args.early_stopper_config, args)

    # Set up clip model   
    with open(args.config_dir) as conf:
        config = yaml.safe_load(conf)[args.clip_model_name]

    clip_model = CLIP(
        **config
    )
    
    # Define phosc model
    phosc_model = create_model(
        model_name=args.phosc_model_name,
        phos_size=args.phos_size,
        phoc_size=args.phoc_size,
        phos_layers=args.phos_layers,
        phoc_layers=args.phoc_layers,
        dropout=args.phosc_dropout
    ).to(device)

    # args.phosc_model = phosc_model

    # Sett phos and phoc language
    set_phos_version(args.phosc_version)
    set_phoc_version(args.phosc_version)

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
        phosc_model=phosc_model,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    validation_set = dset.CompositionDataset(
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
        phosc_model=phosc_model,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

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
        phosc_model=phosc_model,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    optimizer = torch.optim.AdamW(
        clip_model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.98
        ),
        eps=args.epsilon,
        weight_decay=args.weight_decay
    )

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=num_training_steps(train_loader, args.num_epochs, args.batch_size),
        cycle_mult=args.cycle_mult,
        max_lr=args.lr,
        min_lr=0,
        warmup_steps=args.warmup_steps
    )

    image_loader = ImageLoader(ospj(DATA_FOLDER, args.data_dir, args.split_name))

    early_stopping = EarlyStopping(
        save_path=ospj('models', 'trained_clip', args.split_name, args.name),
        patience=args.patience,
        verbose=args.verbose,
        save_every=args.save_every,
        model_arguments=args
    )

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(
            epoch,
            train_loader, 
            clip_model,
            image_loader,
            optimizer,
            lr_scheduler
        )

        validation_loss, validation_acc = validation_one_epoch(
            epoch,
            validation_loader,
            clip_model,
            image_loader
        )

        if early_stopping(validation_loss, clip_model, epoch):
            print('Early stopping')
            print(f'Best model saved at {early_stopping.best_model_path}')
            print(f'Best model validation loss: {early_stopping.best_score}')
            break


if __name__ == '__main__':
    main()
