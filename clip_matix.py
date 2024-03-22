import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

import logging

from PIL import Image

import clip
import os
from os.path import join as ospj

from timm import create_model

from tqdm import tqdm

from flags import DATA_FOLDER, device

from utils import dbe, get_phoscnet, get_test_loader, load_args
from utils.utils import clip_text_features_from_description

from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description
from modules.utils.utils import split_string_into_chunks, get_phosc_description

from modules import models, residualmodels

import numpy as np
import pandas as pd

from torchvision import transforms

from enum import Enum

from parser import phosc_net_argparse, dataset_argparse, early_stopper_argparse, clip_matrix_argparse

from utils.get_dataset import get_training_loader, get_validation_loader, get_test_loader

import argparse

# save_path = ospj('models', 'fine-tuned_clip', split)




def save_matrix_as_csv(matrix, model_save_path, csv_filename="matrix.csv"):
    # Extract the directory from the model save path
    directory = os.path.dirname(model_save_path)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the full path for the CSV file
    csv_path = ospj(directory, csv_filename)

    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()

    # Convert the matrix to a DataFrame and save as CSV
    df = pd.DataFrame(matrix)
    df.to_csv(csv_path, index=False, header=False)

    print(f"Matrix saved as CSV at: {csv_path}")


def process_text_chunks(text_chunks, model, device):
    """Process each text chunk with the model."""
    batch_features = []
    for chunk in text_chunks:
        # Ensure the text chunk is within the context length limit
        tokens = clip.tokenize([chunk]).to(device)

        batch_features.append(tokens)

    return batch_features


def calculate_cos_angle_matrix(vectors):
    n = len(vectors)
    cos_angle_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Convert vectors to PyTorch tensors if they aren't already
            vec_i = vectors[i]
            vec_j = vectors[j]

            # Calculate the dot product of the two vectors
            try:
                dot_product = torch.matmul(vec_i, vec_j)
            except RuntimeError as e:
                dbe(vec_i.shape, vec_j.shape, e)

            # Calculate the magnitudes of the vectors
            magnitude_i = torch.norm(vec_i)
            magnitude_j = torch.norm(vec_j)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_i * magnitude_j)

            # Ensure the cosine value is within the valid range [-1, 1]
            # cos_theta = torch.clamp(cos_theta, -1, 1)

            # Assign the cosine value to the matrix
            cos_angle_matrix[i, j] = cos_theta

    return cos_angle_matrix



def compute_loss_and_accuracy(images_enc, descriptions_enc, image_names, device):
    # Compute loss and accuracy for validation metrics
    image_logits = images_enc @ descriptions_enc.t()
    ground_truth = torch.arange(len(image_logits)).long().to(device)
    loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)

    acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
    acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
    accuracy = (acc_i + acc_t).float() / 2 / len(image_names)

    return loss.item(), accuracy.item()


def clip_process_and_evaluate_batch(image_names, descriptions, model, preprocess, loader, device):
    # Process images
    images = [preprocess(loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
    images = torch.cat(images, dim=0)

    # Precompute embeddings for all descripdtions in the batch
    descriptions_enc = torch.stack([clip_text_features_from_description(description, model) for description in descriptions]).squeeze(1)

    # Encode images using the model
    images_enc = model.encode_image(images)

    # Calculate cosine similarity between each image and text features in the batch
    similarity_matrix = torch.nn.functional.cosine_similarity(images_enc.unsqueeze(1), descriptions_enc.unsqueeze(0), dim=2)
    similarities = similarity_matrix.diag().cpu().tolist()

    return compute_loss_and_accuracy(images_enc, descriptions_enc, image_names, device)


def evaluate_model(model, dataloader, device):
    model.eval()
    similarities = []
    losses = []
    accuracies = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            _, _, _, _, _, _, _, _, image_names, _, descriptions = batch

            clip_process_and_evaluate_batch(image_names, descriptions, model, clip_preprocess, loader, device)


    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_loss, avg_accuracy, similarities


def evaluate_text_embedings(model, dataloader, device, preprocess=clip_preprocess):
    global align_model, align_auto_tokenizer
    model.eval()
    similarities = []
    batch_features_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, desc="Batch Progress"):
            # Unpacking the batch data
            *_, images, _, words = batch

            batch_features_all.append(clip_text_features_from_description(words, model))

    batch_features_all = torch.cat(batch_features_all, dim=0)

    return batch_features_all




def main():
    parser = argparse.ArgumentParser()

    parser = clip_matrix_argparse(parser)
    parser = phosc_net_argparse(parser)
    parser = dataset_argparse(parser)
    parser = early_stopper_argparse(parser)

    args = parser.parse_args()

    load_args(args.clip_matrix_config, args)
    load_args(args.phosc_config, args)
    load_args(args.data_config, args)
    load_args(args.early_stopper_config, args)

    save_path = ospj('models', 'trained_clip', args.split_name)
    model_save_path = ospj(save_path, 'best.pt')
    matrix_save_path = ospj(save_path, 'matrix.csv')
    root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
    image_loader_path = ospj(root_dir, args.split_name)

    phosc_model = get_phoscnet(args)
    test_loader = get_test_loader(args)

    # Preprocessing for CLIP
    clip_preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    loader = ImageLoader(image_loader_path)

    # Load original and fine-tuned CLIP models
    original_clip_model, original_clip_preprocess = clip.load("ViT-B/32", device=device)
    original_clip_model.float()

    # Load fine-tuned clip model
    fine_tuned_clip_model, fine_tuned_clip_preprocess = clip.load("ViT-B/32", device=device)
    fine_tuned_clip_model.float()

    state_dict = torch.load(model_save_path, map_location=device)
    fine_tuned_clip_model.load_state_dict(state_dict)

    batch_features_all = 0

    if args.eval_type == 'image':
        avg_loss, avg_accuracy, similarities = evaluate_model(original_clip_model, test_loader, device)
    elif args.eval_type == 'text':
        batch_features_all = evaluate_text_embedings(original_clip_model, test_loader, device, original_clip_preprocess)

    matrix = calculate_cos_angle_matrix(batch_features_all)

    # Find the minimum and maximum values in the matrix
    min_value = torch.min(matrix).item()
    max_value = torch.max(matrix).item()

    print(f"Minimum value in matrix: {min_value}")
    print(f"Maximum value in matrix: {max_value}")

    save_matrix_as_csv(matrix, matrix_save_path)



if __name__ == '__main__':
    main()
