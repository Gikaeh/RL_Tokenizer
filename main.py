import os

import torch
from torch.utils.data import DataLoader
from time import sleep

from tqdm import tqdm

from policy_network import TransformerPolicyNetwork
from ppo_agent import PPOAgent
from tokenizer_environment import VectorizedTokenizerEnvironment, TokenizerEnvironment
from utils import preprocess_state, load_preprocessed_data, save_model_checkpoint, print_parameters, download_and_preprocess_wikitext, create_directory_structure, compute_efficiency_score
from vocab import Vocabulary
from test import evaluate_word_analogies

# -------------------------------
# -----Hard-Coded Variables------
# -------------------------------

# Names and Data Loading Parameters
DATA_PATH = "data"  # Base path for data files
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
DATASET_NAME = 'wikitext-103-v1'  # Options: 'wikitext-2-v1', 'wikitext-2-raw-v1', 'wikitext-103-v1', 'wikitext-103-raw-v1'
DATA_PERCENTAGE = 1.00 # Percentage of data to use for training

# Evaluation Parameters
STEPS_PER_EPISODE = 15

# Training Parameters
EPOCHS = 5
K_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-6
NUM_ENVIRONMENTS = BATCH_SIZE  # Must match batch size
EMBEDDING_DIM = 128
DROPOUT = 0.1
NUM_HEADS = 8
NUM_LAYERS = 5

# DataLoader Parameters
NUM_WORKERS = 12
PREFETCH_FACTOR = 6

# PPO Parameters
GAMMA = 0.99
EPS_CLIP = 0.2
ACTION_SIZE = 2  # Valid actions (0: Continue, 1: Split)

# Environment Parameters
CONTEXT_SIZE = 15  # Number of context for tokens, N tokens on each side as context
MAX_LENGTH = 300  # Maximum length of tokenized text

# Model Checkpoint and Save Path
MODEL_SAVE_PATH = "models/final/final_policy_model.pth"
MODEL_CHECKPOINT_INTERVAL = 1


def train(agent, envs, vocab, data_loader, validation_data, epochs, device):
    total_steps = 0

    for epoch in range(epochs):
        for batch_num, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            state_list = envs.reset(batch)  # Reset the environments

            for step in range(STEPS_PER_EPISODE):
                state_tensors, attention_masks = preprocess_state(state_list, vocab, device=device)

                # Get Actions
                actions, log_probs, state_values = agent.select_action(state_tensors, attention_masks)
                action = actions.cpu().numpy()

                # Step
                next_states, rewards, dones, token_lengths = envs.step(action.tolist())

                # Store transitions in memory
                agent.store_transitions(states=state_tensors, attention_masks=attention_masks, actions=actions, log_probs=log_probs, state_values=state_values, rewards=rewards, dones=dones)
                state_list = next_states
                total_steps += envs.num_envs

            agent.update_policy()
            print("[Training] Policy updated.")

        if (epoch + 1) % MODEL_CHECKPOINT_INTERVAL == 0:
            save_model_checkpoint(agent, epoch + 1)
            print(f"[Checkpoint] Model checkpoint saved at epoch {epoch + 1}.")

    print("\n[Training] Training complete.")

    # Save the final model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    agent.save_model(MODEL_SAVE_PATH)
    print("[Main] Final model saved.")


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     params = [
#         DATA_PATH, TRAIN_SPLIT, VALIDATION_SPLIT, DATASET_NAME, DATA_PERCENTAGE,
#         STEPS_PER_EPISODE,
#         BATCH_SIZE, NUM_ENVIRONMENTS, NUM_WORKERS, PREFETCH_FACTOR, EPOCHS,
#         EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, ACTION_SIZE, LEARNING_RATE, GAMMA, EPS_CLIP, K_EPOCHS,
#         CONTEXT_SIZE, MAX_LENGTH,
#         MODEL_SAVE_PATH, MODEL_CHECKPOINT_INTERVAL,
#         device
#     ]
#
#     # Print Parameters
#     print_parameters(params)
#
#     # Create necessary directories
#     print("\n[Main] Setting up project directories.")
#     create_directory_structure()
#
#     # Download and preprocess dataset if not already done
#     print(f"\n[Main] Preparing '{DATASET_NAME}' dataset.")
#     download_and_preprocess_wikitext(dataset_name=DATASET_NAME, output_dir=DATA_PATH)
#
#     # Load training and validation data
#     print("\n[Main] Loading training data.")
#     train_data = load_preprocessed_data(split=TRAIN_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
#     train_data = train_data[:int(len(train_data) * DATA_PERCENTAGE)]
#
#     print("[Main] Loading validation data.")
#     validation_data = load_preprocessed_data(split=VALIDATION_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
#     validation_data = validation_data[:int(len(validation_data) * DATA_PERCENTAGE)]
#     print(f"[Main] Loaded {len(train_data)} training examples.")
#     print(f"[Main] Loaded {len(validation_data)} validation examples.")
#
#     print("\n[Main] Initializing vocabulary...")
#     vocab = Vocabulary(train_data)
#     print(f"[Main] Vocabulary Size: {vocab.size}")
#
#     print(f"\n[Main] Preparing DataLoader with batch size: {BATCH_SIZE}")
#     data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR, drop_last=True)
#     print("[Main] DataLoader prepared.")
#
#     # Create vectorized environments
#     print(f"\n[Main] Creating {NUM_ENVIRONMENTS} vectorized environments.")
#     env_fns = [lambda: TokenizerEnvironment(vocab=vocab, context_size=CONTEXT_SIZE) for _ in range(NUM_ENVIRONMENTS)]
#     envs = VectorizedTokenizerEnvironment(env_fns)
#     print("[Main] Vectorized environments created.")
#
#     # Initialize policy network
#     print("\n[Main] Initializing policy network.")
#     policy_network = TransformerPolicyNetwork(vocab_size=vocab.size, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, action_size=ACTION_SIZE, dropout=DROPOUT).to(device)
#     print("[Main] Policy network initialized.")
#
#     # Initialize PPO agent
#     print("\n[Main] Initializing PPO agent.")
#     agent = PPOAgent(policy_network=policy_network, lr=LEARNING_RATE, gamma=GAMMA, eps_clip=EPS_CLIP, k_epochs=K_EPOCHS, device=device)
#     print("[Main] PPO agent initialized.")
#
#     # Compute efficiency score (optional)
#     efficiency_percentage, model_params, dataset_size, gpu_flops_tflops, compute_bottleneck, suggestion = compute_efficiency_score(policy_network, len(train_data))
#     print(f"[Main] Model Parameters: {model_params}")
#     print(f"[Main] Dataset Size: {dataset_size}")
#     print(f"[Main] GPU FLOPs: {gpu_flops_tflops:.2f} TFLOPs")
#     print(f"[Main] Compute Bottleneck: {compute_bottleneck}\n")
#
#     print(f"[Main] Efficiency Score: {efficiency_percentage:.2f}%")
#     print(f"[Main] Suggestion: {suggestion}")
#
#     # Start training
#     print("\n[Main] Starting training process.")
#     sleep(0.1)  # Sleep to allow printing to complete in order
#     train(agent, envs, vocab, data_loader, validation_data, epochs=EPOCHS, device=device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = [
    DATA_PATH, TRAIN_SPLIT, VALIDATION_SPLIT, DATASET_NAME, DATA_PERCENTAGE,
    STEPS_PER_EPISODE,
    BATCH_SIZE, NUM_ENVIRONMENTS, NUM_WORKERS, PREFETCH_FACTOR, EPOCHS,
    EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, ACTION_SIZE, LEARNING_RATE, GAMMA, EPS_CLIP, K_EPOCHS,
    CONTEXT_SIZE, MAX_LENGTH,
    MODEL_SAVE_PATH, MODEL_CHECKPOINT_INTERVAL,
    device
    ]

    # Print Parameters
    print_parameters(params)

    # Create necessary directories
    print("\n[Main] Setting up project directories.")
    create_directory_structure()

    # Download and preprocess dataset if not already done
    print(f"\n[Main] Preparing '{DATASET_NAME}' dataset.")
    download_and_preprocess_wikitext(dataset_name=DATASET_NAME, output_dir=DATA_PATH)

    # Load training and validation data
    print("\n[Main] Loading training data.")
    train_data = load_preprocessed_data(split=TRAIN_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
    train_data = train_data[:int(len(train_data) * DATA_PERCENTAGE)]

    print("[Main] Loading validation data.")
    validation_data = load_preprocessed_data(split=VALIDATION_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
    validation_data = validation_data[:int(len(validation_data) * DATA_PERCENTAGE)]
    print(f"[Main] Loaded {len(train_data)} training examples.")
    print(f"[Main] Loaded {len(validation_data)} validation examples.")

    print("\n[Main] Initializing vocabulary...")
    vocab = Vocabulary(train_data)
    print(f"[Main] Vocabulary Size: {vocab.size}")

    print("\n[Main] Initializing policy network.")
    policy_network = TransformerPolicyNetwork(vocab_size=vocab.size, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, action_size=ACTION_SIZE, dropout=DROPOUT).to(device)
    print("[Main] Policy network initialized.")

    print("\n[Main] Loading model.")
    policy_network.load_state_dict(torch.load(MODEL_SAVE_PATH))
    policy_network.eval()
    print("\n[Main] Model loaded.")

    print(f"[Main] Starting testing on analogies.")
    embeddings = policy_network.embedding.weight.detach().cpu()
    accuracy = evaluate_word_analogies(embeddings, vocab, "data/questions-words.txt", device=device)
    print(f"[Main] Finished testing. Accuracy on word analogies: {accuracy*100:.2f}%")

    # for i in range(len(embeddings)):
    #   print('Word ', i, vocab.idx_to_token.get(i))


if __name__ == "__main__":
    main()