import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm


def print_parameters(params):
    (
        DATA_PATH, TRAIN_SPLIT, VALIDATION_SPLIT, DATASET_NAME, DATA_PERCENTAGE,
        STEPS_PER_EPISODE,
        BATCH_SIZE, NUM_ENVIRONMENTS, NUM_WORKERS, PREFETCH_FACTOR, EPOCHS,
        EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, ACTION_SIZE, LEARNING_RATE, GAMMA, EPS_CLIP, K_EPOCHS,
        CONTEXT_SIZE, MAX_LENGTH,
        MODEL_SAVE_PATH, MODEL_CHECKPOINT_INTERVAL,
        device
    ) = params

    print("\n--- Device Type ---")
    print(f"Device Type: {device}")

    print("\n--- Paths and Data Loading Parameters ---")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"TRAIN_SPLIT: {TRAIN_SPLIT}")
    print(f"VALIDATION_SPLIT: {VALIDATION_SPLIT}")
    print(f"DATASET_NAME: {DATASET_NAME}")
    print(f"DATA_PERCENTAGE: {DATA_PERCENTAGE}")

    print("\n--- Evaluation Parameters ---")
    print(f"STEPS_PER_EPISODE: {STEPS_PER_EPISODE}")

    print("\n--- Training Parameters ---")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_ENVIRONMENTS: {NUM_ENVIRONMENTS}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"PREFETCH_FACTOR: {PREFETCH_FACTOR}")
    print(f"EPOCHS: {EPOCHS}")

    print("\n--- PPO Parameters ---")
    print(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
    print(f"NUM_HEADS: {NUM_HEADS}")
    print(f"NUM_LAYERS: {NUM_LAYERS}")
    print(f"ACTION_SIZE: {ACTION_SIZE}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"GAMMA: {GAMMA}")
    print(f"EPS_CLIP: {EPS_CLIP}")
    print(f"K_EPOCHS: {K_EPOCHS}")

    print("\n--- Environment Parameters ---")
    print(f"CONTEXT_SIZE: {CONTEXT_SIZE}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")

    print("\n--- Model Save and Checkpoint Parameters ---")
    print(f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}")
    print(f"MODEL_CHECKPOINT_INTERVAL: {MODEL_CHECKPOINT_INTERVAL}")


# ChatGPT regex
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
    html_entities = {'&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'"}  # HTML entities to replace
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    text = text.encode('ascii', 'ignore').decode('utf-8')  # Remove non-ASCII characters
    text = re.sub(r'[^a-z0-9\s.,!?\'\"()\-]', '', text)  # Remove special characters except for punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def download_and_preprocess_wikitext(dataset_name, output_dir = "data"):
    dataset_version = f'{dataset_name}'
    splits = ['train', 'validation', 'test']
    split_files = {split: os.path.join(output_dir, f"{dataset_version}-{split}.json") for split in splits}

    if all([os.path.isfile(path) for path in split_files.values()]):
        print(f"[Utils] Preprocessed {dataset_version} data already exists. Skipping download and preprocessing.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Utils] Loading {dataset_version} dataset.")
    dataset = load_dataset('wikitext', dataset_version)
    print(f"[Utils] Loaded {dataset_version} dataset.")

    # Preprocess data
    for split in splits:
        split_data = dataset[split]
        split_file = split_files[split]
        if os.path.isfile(split_file):
            print(f"[Utils] {split.capitalize()} split already exists at {split_file}. Skipping.")
            continue

        print(f"[Utils] Processing and saving {split} split.")
        with open(split_file, 'w', encoding='utf-8') as f:
            for example in tqdm(split_data, desc=f"Processing {split}"):
                text = preprocess_text(example['text'])
                if text:
                    json.dump({"text": text}, f)
                    f.write('\n')

    print(f"[Utils] {dataset_version} dataset downloaded and preprocessed.")


def load_preprocessed_data(split, output_dir, dataset_name):
    split_file = os.path.join(output_dir, f"{dataset_name}-{split}.json")
    if not os.path.isfile(split_file):
        print(f"[Utils] Error: Preprocessed split file {split_file} not found.")
        raise FileNotFoundError(f"Preprocessed split file {split_file} not found.")

    print(f"[Utils] Loading {split} split from {split_file}...")
    texts = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {split}"):
            data = json.loads(line)
            text = data.get('text', '').strip()
            if text:
                texts.append(text)
    print(f"[Utils] Loaded {len(texts)} samples from {split} split.")
    return texts


# Process the padded text sequence batches and attention masks
def preprocess_state(states, vocab, device):
    combined_texts = []
    for state in states:
        current_context = state['current_context']
        previous_token = state['previous_token']
        combined = f"{previous_token}{current_context}".strip() # Combine previous token and context
        combined_texts.append(combined)

    tokens_list = [vocab.tokenize(text) for text in combined_texts] # Initialize combined texts
    encoded_list = [vocab.encode(tokens) for tokens in tokens_list] # Encode tokens to indices

    # Get seqeuence lengths
    lengths = [len(encoded) for encoded in encoded_list]
    max_length = max(lengths)

    # Create padded sequences and attention masks
    pad_idx = vocab.token_to_idx["<pad>"]  # Get padding index, should be 0
    padded_encoded = torch.full((len(encoded_list), max_length), pad_idx, dtype=torch.long) # Initialize tensor with padding index
    attention_masks = torch.zeros((len(encoded_list), max_length), dtype=torch.bool) # Initialize tensor attention masks

    # Iterate over encoded sequences and lengths
    for i, encoded in enumerate(encoded_list):
        length = lengths[i]
        padded_encoded[i, :length] = torch.tensor(encoded, dtype=torch.long)
        attention_masks[i, :length] = True # Mark the valid positions in the attention mask

    # Move tensors to device
    padded_encoded = padded_encoded.to(device)
    attention_masks = attention_masks.to(device)

    return padded_encoded, attention_masks


def preprocess_single_state(state, vocab, max_length, device):
    current_context = state['current_context'] # Get current context
    previous_token = state['previous_token'] # Get previous token
    combined = f"{previous_token}{current_context}".strip() # Combine previous token and context
    tokens = vocab.tokenize(combined) # Tokenize the combined text

    # Check for empty tokens
    if not tokens:
        tokens = ['<unk>'] # Replace empty tokens with unknown token if empty

    encoded = vocab.encode(tokens)

    # Pad / Truncate sequences
    pad_idx = vocab.token_to_idx["<pad>"]
    if len(encoded) < max_length:
        padded_encoded = encoded + [pad_idx] * (max_length - len(encoded)) # Pad the sequence with padding index to match max length
        attention_mask = [1] * len(encoded) + [0] * (max_length - len(encoded)) # Create attention mask for valid positions
    else: # Sequence is longer than max length
        padded_encoded = encoded[:max_length] # Truncate the sequence to match max length
        attention_mask = [1] * max_length # Create attention mask for valid positions

    # Convert to tensors
    encoded_tensor = torch.tensor([padded_encoded], dtype=torch.long, device=device)  # Shape: (1, max_length)
    attention_mask = torch.tensor([attention_mask], dtype=torch.bool, device=device)  # Shape: (1, max_length)

    return encoded_tensor, attention_mask


def save_model_checkpoint(agent, epoch, checkpoint_dir = "models/checkpoint"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    agent.save_model(checkpoint_path)
    print(f"[Utils] Model checkpoint saved at {checkpoint_path}")


def create_directory_structure():
    directories = ['data', 'models', 'models/checkpoint', 'models/final', 'tokenizers', 'tokenizers/bpe', 'tokenizers/RL']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(f"[Utils] Created necessary directories.")


def compute_efficiency_score(model, dataset_size):
    model_params = sum(p.numel() for p in model.parameters())
    gpu_flops_tflops = 82.58  # RTX 4090 FLOPs in TFLOPs

    # Chinchilla law - Optimal ratio of parameters to dataset size (~0.5 is ideal)
    optimal_params_tokens_ratio = 0.5
    current_ratio = model_params / dataset_size
    model_type_scaling = 1.2 # Transformer models are more efficient than other architectures
    adjusted_flops = gpu_flops_tflops * model_type_scaling * 1e12

    # Calculate efficiency score based on the optimal parameter-token ratio
    efficiency_score = (optimal_params_tokens_ratio / current_ratio) * (adjusted_flops / (model_params * dataset_size))

    # Diminishing returns: penalize for extreme ratios
    if current_ratio < 0.3:
        efficiency_score *= 0.85  # Penalize smaller models for large datasets
    elif current_ratio > 1.0:
        efficiency_score *= 0.75  # Penalize large models for smaller datasets

    # Convert the efficiency score to a percentage (out of 100%)
    efficiency_percentage = min(efficiency_score * 100, 100)

    if current_ratio < optimal_params_tokens_ratio:
        suggestion = "Increase model size or reduce dataset size."
    elif current_ratio > optimal_params_tokens_ratio:
        suggestion = "Increase dataset size to match model size."
    else:
        suggestion = "Balanced model and dataset."

    # Check for GPU compute bottleneck
    compute_bottleneck = adjusted_flops < (model_params * dataset_size)
    if compute_bottleneck:
        suggestion += " Your compute power is a limiting factor."

    return efficiency_percentage, model_params, dataset_size, gpu_flops_tflops, compute_bottleneck, suggestion



