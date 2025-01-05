import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Core Configuration
MODEL_ARCHIVE = 'Qwen/Qwen-1_8B-chat'  # Model identifier for the Qwen model
DEVICE_TYPE = 'cuda'  # Device type for torch (GPU acceleration)
MAX_GENERATION_LENGTH = 512  # Maximum length for text generation
TRAIN_SAMPLES = 32  # Number of training samples to use
TEST_SAMPLES = 32  # Number of test samples to use

# Initialize the Transformer Model
def initialize_model(model_path: str, device: str):
    """Initializes and configures a transformer model for analysis."""
    transformer = HookedTransformer.from_pretrained_no_processing(
        model_path,
        device=device,
        dtype=torch.float16,
        default_padding_side='left',
        fp16=True
    )
    transformer.tokenizer.padding_side = 'left'
    transformer.tokenizer.pad_token = '<|extra_0|>'
    return transformer

# Load and Prepare Datasets
def prepare_datasets():
    """Loads harmful and harmless datasets from Hugging Face datasets."""
    harmful_dataset = load_dataset('mlabonne/harmful_behaviors')
    harmless_dataset = load_dataset('mlabonne/harmless_alpaca')
    harmful_train = harmful_dataset['train']['text'][:TRAIN_SAMPLES]
    harmless_train = harmless_dataset['train']['text'][:TRAIN_SAMPLES]
    return harmful_train, harmless_train

# PCA Analysis Function
def perform_pca(activations):
    """Performs PCA on the provided activations and returns the principal components."""
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(activations)
    explained_variance = pca.explained_variance_ratio_
    return transformed_data, explained_variance

# Plotting Function
def plot_activations(harmful_data, harmless_data, explained_variance, title):
    """Plots the PCA-transformed activations to visualize differences in model behavior."""
    plt.figure(figsize=(12, 6))
    plt.scatter(harmful_data[:, 0], harmful_data[:, 1], label='Harmful', color='red', alpha=0.5)
    plt.scatter(harmless_data[:, 0], harmless_data[:, 1], label='Harmless', color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% Variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% Variance)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Execution Block
if __name__ == "__main__":
    # Initialize model and prepare data
    model = initialize_model(MODEL_ARCHIVE, DEVICE_TYPE)
    harmful_train, harmless_train = prepare_datasets()

    # Example activation extraction method (hypothetical, depends on actual model and data structure)
    harmful_activations = np.random.randn(len(harmful_train), 100)  # Simulated data
    harmless_activations = np.random.randn(len(harmless_train), 100)  # Simulated data

    # Perform PCA on both datasets
    harmful_pca, harm_variance = perform_pca(harmful_activations)
    harmless_pca, harmless_variance = perform_pca(harmless_activations)

    # Plot results
    plot_activations(harmful_pca, harmless_pca, harm_variance, 'PCA of Model Activations - Harmful vs. Harmless')
