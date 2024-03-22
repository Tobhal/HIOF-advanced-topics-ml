import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import math
# from transformers import CLIPTextConfig, CLIPVisionConfig, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

import logging

import clip
import os
from os.path import join as ospj

from timm import create_model

from tqdm import tqdm

from flags import DATA_FOLDER, device

from utils.dbe import dbe
from utils.load_args import load_args
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader
from utils.early_stopping import EarlyStopping

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description, gen_shape_description_simple

from modules import models, residualmodels

from typing import Callable
from typing import Tuple

from enum import Enum

from train_clip.utils.clip_utils import gen_word_objs_embeddings

import random

from parser import phosc_net_argparse, dataset_argparse, early_stopper_argparse, clip_fine_tune_argparse

from utils.get_dataset import get_training_loader, get_validation_loader, get_test_loader

import argparse

# Initialize logging
def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info("Initialized logging")


# Function to check if the model save path's directory exists
def verify_model_save_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        print(f"Model save directory does not exist, creating: {directory}")
        os.makedirs(directory)


def custom_loss(image_features, text_features):
    # Assuming image_features and text_features are normalized
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
    loss = torch.mean(1 - similarity)  # Penalize high similarity
    return loss


def normalize_features(features):
    return features / features.norm(dim=1, keepdim=True)


# Cross entropy helper function
def cross_entropy(logits, axis):
    logprobs = torch.log_softmax(logits, axis=axis)
    nll = torch.diag(logprobs)
    ce = -torch.mean(nll)
    return ce


def custom_loss_same_class(anchor_image_features, positive_text_features):
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(positive_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = -((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_loss_different_class(anchor_image_features, negative_text_features):
    # Ensure features are normalized
    image_features = F.normalize(anchor_image_features, dim=1)
    text_features = F.normalize(negative_text_features, dim=1)

    # Calculate similarity
    similarity = torch.matmul(image_features, text_features.T)

    # Compute CLIP loss
    loss = 1 - ((cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2)
    return loss


def custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features, margin=1.0):
    # Calculate triplet loss
    loss = F.triplet_margin_loss(
        anchor_image_features,
        positive_text_features,
        negative_text_features,
        margin=margin
    )
    return loss

class Loss_method(Enum):
    DIFFRENT_SAME = 1
    CUSTOM_TRIPLET_LOSS = 2


def calc_loss(anchor_image_features, positive_text_features, negative_text_features, is_same_class: bool, loss_method: Loss_method):
    if loss_method == Loss_method.DIFFRENT_SAME:
        if is_same_class:
            return custom_loss_same_class(anchor_image_features, positive_text_features)
        else:
            return custom_loss_different_class(anchor_image_features, negative_text_features)

    elif loss_method == Loss_method.CUSTOM_TRIPLET_LOSS:
        return custom_triplet_loss(anchor_image_features, positive_text_features, negative_text_features)

def create_learning_rate_fn(
    optimizer,
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    linear=False
):
    """Returns a PyTorch learning rate scheduler."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if linear:
            return max(
                0.0, float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps))
            )
        else:  # Cosine decay
            return 0.5 * (1 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_train_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def adaptive_grad_clip(parameters, clip_factor, eps):
    for p in parameters:
        if p.grad is not None:
            grad_norm = p.grad.norm()
            max_norm = clip_factor / (eps + grad_norm)
            p.grad.data.clamp_(-max_norm, max_norm)


def train_step_batch(images, text_feat, batch_size, clip_model, optimizer=None, lr_scheduler=None):
    # Transform setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Process images
    images = torch.stack([transform(img) for img in images])
    # If your images are already in batches, you might adjust this part accordingly
    image_emb = torch.chunk(images, math.ceil(len(images) / batch_size))

    # Normalize image embeddings after encoding
    ims = [F.normalize(clip_model.encode_image(batch), dim=1) for batch in image_emb]
    ims = torch.cat(ims)

    # Normalize text features after averaging across the token dimension
    txt = F.normalize(text_feat, dim=1)

    # Compute similarity scores between images and texts
    image_logits = ims @ txt.t() * clip_model.logit_scale.exp()

    # Compute triplet loss
    positive_pairs = image_logits.diag()
    negative_pairs = image_logits[~torch.eye(image_logits.size(0)).bool()].reshape(image_logits.size(0), -1).max(dim=1)[0]
    margin = 1.0  # You can adjust the margin as needed
    total_loss = F.relu(positive_pairs - negative_pairs + margin).mean()

    optimizer.zero_grad()
    total_loss.backward()

    if optimizer:
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    # Clamp the logit scale
    clip_model.logit_scale.data = clip_model.logit_scale.data.clamp(-np.log(100), np.log(100))

    return total_loss.item()


def train_one_epoch(epoch: int, train_loader, clip_model, image_loader, optimizer=None, lr_scheduler=None) -> float:
    clip_model.train()
    running_loss = 0

    train_bar = tqdm(train_loader, desc=f'Fine-tuning epoch {epoch + 1}')
    for batch in train_bar:
        optimizer.zero_grad()

        *_, image_names, _, words = batch

        images = [image_loader(image) for image in tqdm(image_names, position=1, desc='Processing Images', leave=False)]
        # emb_descriptions = [gen_word_objs_embeddings(description, clip_model) for description in tqdm(descriptions, position=1, desc='Generating Embeddings', leave=False)]

        # emb_descriptions = torch.stack(emb_descriptions)

        temp_loss = train_step_batch(images, words, 32, clip_model, optimizer, lr_scheduler)

        running_loss += temp_loss

    loss = running_loss / len(train_loader)

    if optimizer:
        optimizer.step()

    if lr_scheduler:
        lr_scheduler.step()

    return loss


def validate_one_epoch(epoch, val_loader, clip_model, clip_preprocess, loader, scheduler=None):
    global device

    clip_model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    total_samples = 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        for batch in val_bar:
            *_, image_names, _, descriptions = batch

            # Preprocess images
            images = [clip_preprocess(loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
            images = torch.cat(images, dim=0)

            # Encode images
            images_enc = clip_model.encode_image(images)

            # Process and encode descriptions
            descriptions_enc = torch.stack([gen_word_objs_embeddings(description, clip_model) for description in descriptions]).squeeze(1)

            # Compute similarity scores between images and texts
            image_logits = images_enc @ descriptions_enc.t()
            ground_truth = torch.arange(len(image_logits)).long().to(device)

            # Calculate loss
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            val_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Calculate accuracy
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            val_accuracy += (acc_i + acc_t).float().item() / 2

            val_bar.set_description(f"Validation Epoch {epoch + 1} Loss: {loss.item():.4f}")

        # Compute average loss and accuracy
        val_loss /= total_samples
        val_accuracy /= total_samples

    # Logging validation loss and accuracy
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    if scheduler:
        scheduler.step(val_loss)  # Assuming the scheduler uses validation loss to adjust LR

    return val_loss, val_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser = aling_fine_tune_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)

    args = parser.parse_args()

    load_args(args.clip_fine_tune_config, args)
    load_args(args.phosc_config, args)
    load_args(args.data_config, args)
    load_args(args.early_stopper_config, args)

    # Define patchs
    root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
    image_loader_path = ospj(root_dir, args.split_name)

    root_model_path = ospj('models', 'fine-tuned_clip', args.split_name)
    log_file_path = ospj(root_model_path, 'training_log.log')
    model_save_path = ospj(root_model_path, 'simple', 'model.pth')

    verify_model_save_path(model_save_path)

    # Set up models
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

    # Load phosc model
    phosc_model = create_model(
        model_name=args.phosc_model_name,
        phos_size=args.phos_size,
        phoc_size=args.phoc_size,
        phos_layers=args.phos_layers,
        phoc_layers=args.phoc_layers,
        dropout=args.phosc_dropout
    ).to(device)

    # Sett phos and phoc language
    set_phos_version('ben')
    set_phoc_version('ben')

    # Setup datasett
    train_loader, train_set = get_training_loader(args)
    validation_loader, _ = get_validation_loader(args)

    # Define image loader
    loader = ImageLoader(ospj(DATA_FOLDER, args.data_dir, args.split_name))

    early_stopping = EarlyStopping(
        save_path=ospj('models', 'fine-tuned_clip', args.split_name, args.name),
        patience=args.stop_patience,
        verbose=args.verbose,
        save_every=args.save_every,
        model_arguments=args
    )

    # Fine-tuning
    optimizer = torch.optim.RMSprop(
        clip_model.parameters(),
        lr=args.lr
    )

    decay_lr_schedule_fn = create_learning_rate_fn(
        optimizer,
        len(train_set),
        args.batch_size,
        args.epochs,
        args.warmup_steps,
        args.lr,
        linear=False,  # set False to activate cosine annealing
    )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            epoch,
            train_loader,
            clip_model,
            loader,
            optimizer,
            decay_lr_schedule_fn
        )

        val_loss, val_acc = validate_one_epoch(
            epoch,
            validation_loader,
            clip_model,
            clip_preprocess,
            loader,
            decay_lr_schedule_fn
        )

        if early_stopping(val_loss, clip_model, epoch):
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Ctrl-C exit')
