from typing_extensions import Any, Literal
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import logging

from PIL import Image

import torch.nn.functional as F

import clip
import os
from os.path import join as ospj

from timm import create_model


from tqdm import tqdm

from flags import DATA_FOLDER, device

from utils.dbe import dbe
from data import dataset_bengali as dset
from data.dataset_bengali import ImageLoader

from modules.utils import set_phos_version, set_phoc_version, gen_shape_description

from modules import models, residualmodels
from modules.utils.utils import get_phosc_description

import numpy as np

from utils.dbe import dbe
from utils.utils import clip_text_features_from_description

from torchvision import transforms
from typing import List, Tuple

from modules.utils.utils import split_string_into_chunks

# align
from transformers import AlignProcessor, AlignModel, AutoTokenizer
from enum import Enum

split = 'Fold0_use_50'
use_augmented = False

# czsl/models/fine-tuned_clip/Fold0_use/simple/18/best.pt
finetuned_model_save_path = ospj('models', 'trained_clip', split, 'bengali_words', '1', 'best.pt')
trained_model_save_path = ospj('models', 'trained_clip', split, 'super_aug', '2', 'best.pt')
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
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

# Load original and fine-tuned CLIP models
original_clip_model, original_clip_preprocess = clip.load("ViT-B/32", device=device)
original_clip_model.float()

# Load fine-tuned clip model
fine_tuned_clip_model, fine_tuned_clip_preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_clip_model.float()

fine_tuned_state_dict = torch.load(finetuned_model_save_path, map_location=device)
fine_tuned_clip_model.load_state_dict(fine_tuned_state_dict)

# Preprocessing for CLIP
clip_preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

loader = ImageLoader(image_loader_path)


def calculate_cos_angle_matrix(vector_1, vector_2):
    """
    Calculate the cosine of the angle between two vectors

    args:
        vector_1: First vector
        vector_2: Second vector

    returns:
        Cosine of the angle between the two vectors
    """
    # Ensure the vectors are PyTorch tensors and flatten them if they are 2D
    vector_1 = torch.tensor(vector_1).flatten()
    vector_2 = torch.tensor(vector_2).flatten()

    vectors = [vector_1, vector_2]
    n = len(vectors)
    cos_angle_matrix = torch.zeros((n, n))

    # Iterate over the vectors
    for i in range(n):
        for j in range(n):
            # Calculate the dot product of the two vectors
            dot_product = torch.matmul(vectors[i], vectors[j])

            # Calculate the magnitudes of the vectors
            magnitude_a = torch.norm(vectors[i])
            magnitude_b = torch.norm(vectors[j])

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude_a * magnitude_b)

            # Ensure the cosine value is within the valid range [-1, 1]
            cos_theta = torch.clamp(cos_theta, -1, 1)

            # Assign the cosine value to the matrix
            cos_angle_matrix[i, j] = cos_theta

    return cos_angle_matrix


def clip_preprocess_and_encode(image_names, words, clip_model, transform, loader, device):
    """
    Preprocess images and encode them using the CLIP model

    Args:
        image_names: List of image file paths
        words: List of text descriptions
        clip_model: CLIP model
        transform: Image transformation
        loader: Image loader
        device: Device to run the model on

    Returns:
        Tuple of image and text embeddings
    """

    # Preprocess and encode images
    images = [transform(loader(img_name)).unsqueeze(0).to(device) for img_name in image_names]
    # image = transform(loader(image_names)).unsqueeze(0).to(device)
    images = torch.cat(images, dim=0)
    images_features = clip_model.encode_image(images)

    # Precompute embeddings for all descriptions in the batch
    text_features = words
    # text_features = clip_text_features_from_description(words, clip_model)

    images_features /= images_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores between images and texts
    similarity = (images_features @ text_features.t()).softmax(dim=1)

    values, indices = similarity[0].topk(5)

    # Compute cosine similarity between image and text embeddings
    # similarity_matrix = torch.nn.functional.cosine_similarity(image_logits, ground_truth, dim=0)

    return values, indices


def evaluate_model(model, dataloader, device, loader) -> np.floating[Any]:
    """
    Evaluate the model on the given dataset.

    args:
        model: The CLIP model to evaluate.
        dataloader: The dataloader for the dataset.
        device: The device to run the evaluation on.
        loader: The image loader for the dataset.

    returns:
        The average similarity between the image and text embeddings.
    """
    model.eval()
    batch_similarities_values = []
    batch_similarities_indicies = []

    # Define the transformation for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Evaluate the model on the dataset
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            *_, image_names, _, words = batch

            # Generate embeddings for the text descriptions
            w = torch.stack([clip_text_features_from_description(word, model) for word in tqdm(words, position=1, desc='Generating Embeddings', leave=False)]).squeeze(1)

            # Preprocess and encode the images
            values, indicies = clip_preprocess_and_encode(image_names, w, model, transform, loader, device)

            # Append the similarity values and indicies to the batch lists
            batch_similarities_values.append(values)
            batch_similarities_indicies.append(indicies)

    flat_similarities = [item for sublist in batch_similarities_values for item in sublist]

    # Compute average similarities for same and different classes
    avg_same_class_similarity = np.mean(flat_similarities)

    return avg_same_class_similarity


def summarize_results(*args: Tuple[str, np.floating[Any]]):
    """
    Summarize the results of the evaluation by computing the average similarity scores for each model and determining
    which model performs better for same-class and different-class pairs.

    Args:
        args: A tuple of model names and their corresponding average similarity scores for same-class pairs.
    """
    # Extract the model names and average similarity scores
    model_names = [arg[0] for arg in args]
    avg_same_class_similarities = [arg[1] for arg in args]

    # Determine which model performs better for same-class pairs
    best_same_class_model = model_names[np.argmax(avg_same_class_similarities)]

    # Print the results
    print(f"Average Similarity Scores:")
    for model_name, avg_same_class_similarity in zip(model_names, avg_same_class_similarities):
        print(f"{model_name}: {avg_same_class_similarity:.5f}")

    print(f"\nBest Model for Same-Class Pairs: {best_same_class_model}")
    print(f"Worst Model for Same-Class Pairs: {model_names[1 - model_names.index(best_same_class_model)]}")


# Evaluate both models
original_distances_same = evaluate_model(original_clip_model, test_loader, device, loader)
fine_tuned_distances_same = evaluate_model(fine_tuned_clip_model, test_loader, device, loader)

# Compare and summarize results
summarize_results(("Original CLIP", original_distances_same), ("Fine-Tuned CLIP", fine_tuned_distances_same))
