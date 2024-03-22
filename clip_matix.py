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

from utils.dbe import dbe
from utils.utils import clip_text_features_from_description

from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description
from modules.utils.utils import split_string_into_chunks, get_phosc_description

from modules import models, residualmodels

import numpy as np
import pandas as pd

from torchvision import transforms

from transformers import AlignProcessor, AlignModel, AutoTokenizer, AutoProcessor, AlignTextModel, AlignConfig
from enum import Enum

split = 'Fold0_use_50'
use_augmented = False

# align model
align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
align_model = AlignModel.from_pretrained("kakaobrain/align-base")
align_auto_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
align_auto_processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

# align_text_model = AlignTextModel.from_pretrained("kakaobrain/align-base")

# Create a new configuration with a larger maximum sequence length
# config = AlignConfig.from_pretrained("kakaobrain/align-base", max_position_embeddings=2048)

# Create a new model with the updated configuration
# align_text_model = AlignTextModel(config)

# save_path = ospj('models', 'fine-tuned_clip', split)
save_path = ospj('models', 'trained_clip', split)
model_save_path = ospj(save_path, 'best.pt')
matrix_save_path = ospj(save_path, 'matrix.csv')
root_dir = ospj(DATA_FOLDER, "BengaliWords", "BengaliWords_CroppedVersion_Folds")
image_loader_path = ospj(root_dir, split)

# Define phosc model
phosc_model = create_model(
    model_name='ResNet18Phosc',
    phos_size=195,
    phoc_size=1200,
    phos_layers=1,
    phoc_layers=1,
    dropout=0.5
).to(device)

# Sett phos and phoc language
set_phos_version('ben')
set_phoc_version('ben')

# Assuming you have the necessary imports and initializations done (like dset, phosc_model, etc.)
testset = dset.CompositionDataset(
    root=root_dir,
    phase='train',
    split=split,
    # phase='test'
    # split='fold_0_new',
    model='resnet18',
    num_negs=1,
    pair_dropout=0.5,
    update_features=False,
    train_only=True,
    open_world=True,
    augmented=use_augmented,
    add_original_data=True,
    # phosc_model=phosc_model,
    # clip_model=clip_model
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# Load original and fine-tuned CLIP models
original_clip_model, original_clip_preprocess = clip.load("ViT-B/32", device=device)
original_clip_model.float()

"""
# Load fine-tuned clip model
fine_tuned_clip_model, fine_tuned_clip_preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_clip_model.float()

state_dict = torch.load(model_save_path, map_location=device)
fine_tuned_clip_model.load_state_dict(state_dict)
"""

# Preprocessing for CLIP
clip_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

loader = ImageLoader(image_loader_path)


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


class ModelType(Enum):
    CLIP = "CLIP"
    ALIGN = "ALIGN"


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


def align_process_and_evaluate_batch(image_names, descriptions, model, transform, device):
    images = [loader(img_name) for img_name in image_names]
    losses = []
    accuracy = []

    for image, description in zip(images, descriptions):
        processor_inputs = align_auto_processor(image=image, return_tensors="pt")
        text_input = align_auto_tokenizer(description, return_tensors="pt")

        image_features = model.get_image_features(**processor_inputs)
        text_features = model.get_text_features(**text_input)

        dbe(image_features.shape, text_features.shape)

        # Compute similarity scores between images and texts
        loss, acc = compute_loss_and_accuracy(image_features, text_features, image_names, device)

        losses.append(loss)
        accuracy.append(acc)

    return losses, accuracy


def evaluate_model_batch(model, dataloader, device, model_type: ModelType):
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

            if model_type == ModelType.CLIP:
                clip_process_and_evaluate_batch(image_names, descriptions, model, clip_preprocess, loader, device)
            elif model_type == ModelType.ALIGN:
                align_process_and_evaluate_batch(image_names, descriptions, model, transform, device)


    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_loss, avg_accuracy, similarities


def evaluate_text_embedings(model, dataloader, device, preprocess=clip_preprocess, model_type: ModelType = ModelType.CLIP):
    global align_model, align_auto_tokenizer
    model.eval()
    similarities = []
    batch_features_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, position=0, desc="Batch Progress"):
            # Unpacking the batch data
            *_, images, _, words = batch

            if model_type == ModelType.CLIP:
                batch_features_all.append(clip_text_features_from_description(words, model))
            elif model_type == ModelType.ALIGN:
                for image, word in zip(images, words):
                    image = loader(image)
                    description = get_phosc_description(word)

                    inputs = align_auto_tokenizer(description, padding=True, return_tensors="pt")

                    text_features = model.get_text_features(**inputs)

                    batch_features_all.append(text_features)

    batch_features_all = torch.cat(batch_features_all, dim=0)

    return batch_features_all


if __name__ == '__main__':
    # similarities, batch_features_all = evaluate_model(original_clip_model, test_loader, device, original_clip_preprocess)
    # similarities, batch_features_all = evaluate_model(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_preprocess)

    # batch_features_all = evaluate_text_embedings(original_clip_model, test_loader, device, original_clip_preprocess)
    # batch_features_all = evaluate_text_embedings(fine_tuned_clip_model, test_loader, device, fine_tuned_clip_preprocess)
    batch_features_all = evaluate_text_embedings(align_model, test_loader, device, align_processor, ModelType.ALIGN)

    matrix = calculate_cos_angle_matrix(batch_features_all)

    # Find the minimum and maximum values in the matrix
    min_value = torch.min(matrix).item()
    max_value = torch.max(matrix).item()

    print(f"Minimum value in matrix: {min_value}")
    print(f"Maximum value in matrix: {max_value}")

    save_matrix_as_csv(matrix, matrix_save_path)
