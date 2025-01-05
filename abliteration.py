import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from torch import Tensor
from tqdm import tqdm
from functools import partial
from einops import einsum
from jaxtyping import Float, Int
from sklearn.metrics import accuracy_score
from typing import List, Callable

# Core Configuration
MODEL_ARCHIVE = 'Qwen/Qwen-1_8B-chat'  # Model identifier
DEVICE_TYPE = 'cuda'  # Device type for torch (GPU acceleration)
MAX_GENERATION_LENGTH = 512  # Maximum text generation length
TRAIN_SAMPLES = 32  # Number of training samples
TEST_SAMPLES = 32  # Number of testing samples
BATCH_PROCESS_SIZE = 4  # Batch processing size for model operations

# Initialize the Transformer Model
def initialize_model(model_path: str, device: str):
    """Initializes and returns a HookedTransformer model loaded with specific configurations."""
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

# Instruction Template for Chat
CHAT_TEMPLATE = """<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""

# Generate Encoded Tokens
def encode_instructions(tokenizer, queries: List[str]) -> Tensor:
    """Encodes user queries using a predefined chat template for processing."""
    formatted_queries = [CHAT_TEMPLATE.format(query=q) for q in queries]
    return tokenizer(formatted_queries, padding=True, truncation=False, return_tensors="pt").input_ids

# Load and Prepare Datasets
def load_instruction_datasets(dataset_name: str):
    """Loads datasets for harmful and harmless behaviors from Hugging Face."""
    raw_data = load_dataset(dataset_name)
    return raw_data['train']['text'], raw_data['test']['text']

def prepare_datasets():
    """Prepares training and testing datasets by loading harmful and harmless behaviors."""
    harm_train, harm_test = load_instruction_datasets('mlabonne/harmful_behaviors')
    harmless_train, harmless_test = load_instruction_datasets('mlabonne/harmless_alpaca')
    return harm_train, harm_test, harmless_train, harmless_test

# Abliteration Hooks and Functions
def compute_intervention_vector(model, train_data, harmlessness_data, layer, position):
    """Computes and returns the intervention vector to neutralize censorship mechanisms."""
    # Specific computation logic here (e.g., calculating the difference in activations)
    pass

def intervention_hook(activation, hook, direction_vector):
    """Modifies activations at runtime to suppress unwanted censorship behavior."""
    projection = einsum(activation, direction_vector, '... d, d -> ...') * direction_vector
    return activation - projection

def calculate_accuracy(results, ground_truth):
    """Calculates the accuracy of the model's predictions compared to ground truth labels."""
    predictions = [classify_response(res) for res in results]
    return accuracy_score(ground_truth, predictions)

# Main execution workflow
if __name__ == "__main__":
    transformer_model = initialize_model(MODEL_ARCHIVE, DEVICE_TYPE)
    harmful_train, harmful_test, harmless_train, harmless_test = prepare_datasets()

    # Define layer and position for ablation
    ablation_layer = 14
    attention_position = -1

    # Compute intervention vector
    intervention_vector = compute_intervention_vector(transformer_model, harmful_train, harmless_train, ablation_layer, attention_position)

    # Apply hooks for abliteration
    ablation_function = partial(intervention_hook, direction_vector=intervention_vector)
    activation_hooks = [
        (utils.get_act_name(act, layer), ablation_function)
        for layer in range(transformer_model.cfg.n_layers)
        for act in ['resid_pre', 'resid_mid', 'resid_post']
    ]

    # Generate responses and compare censored vs. uncensored results
    censored_results = batch_generate_responses(
        transformer_model, harmful_test[:TEST_SAMPLES], partial(encode_instructions, transformer_model.tokenizer), []
    )
    uncensored_results = batch_generate_responses(
        transformer_model, harmful_test[:TEST_SAMPLES], partial(encode_instructions, transformer_model.tokenizer), activation_hooks
    )

    # Calculate accuracy for both models
    harmful_test_truth = ["harmless" if "safe" in text else "harmful" for text in harmful_test[:TEST_SAMPLES]]
    censored_accuracy = calculate_accuracy(censored_results, harmful_test_truth)
    uncensored_accuracy = calculate_accuracy(uncensored_results, harmful_test_truth)

    # Print results
    for index, instruction in enumerate(harmful_test[:TEST_SAMPLES]):
        print("=" * 80)
        print(f"Instruction {index + 1}: {instruction.strip()}")
        print("-" * 80)
        print("Censored:", censored_results[index].strip())
        print("Uncensored:", uncensored_results[index].strip())
        print("=" * 80)

    print(f"Censored Model Accuracy: {censored_accuracy * 100:.2f}%")
    print(f"Uncensored Model Accuracy: {uncensored_accuracy * 100:.2f}%")
