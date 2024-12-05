import os

import optuna
import torch
from torch.utils.data import DataLoader
from time import sleep

from tqdm import tqdm

from policy_network import TransformerPolicyNetwork
from ppo_agent import PPOAgent
from tokenizer_environment import VectorizedTokenizerEnvironment, TokenizerEnvironment
from utils import preprocess_state, load_preprocessed_data, save_model_checkpoint, print_parameters, download_and_preprocess_wikitext, create_directory_structure, compute_efficiency_score, download_and_preprocess_giga, load_preprocessed_giga
from vocab import Vocabulary
from test import evaluate_word_analogies
from alt_model import train_cbow, train_skipgram, get_word2vec_embeddings


# -------------------------------
# -----Hard-Coded Variables------
# -------------------------------

# Names and Data Loading Parameters
DATA_PATH = "data"  # Base path for data files
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
DATASET_NAME = 'wikitext-103-v1'  # Options: 'wikitext-2-v1', 'wikitext-2-raw-v1', 'wikitext-103-v1', 'wikitext-103-raw-v1'
DATA_PERCENTAGE = 1 # Percentage of data to use for training

# Environment Parameters
STEPS_PER_EPISODE = 100
CONTEXT_SIZE = 25  # Number of context for tokens, N tokens on each side as context

# Training Parameters
EPOCHS = 3
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
EPS_CLIP = 0.1
ACTION_SIZE = 2  # Valid actions (0: Continue, 1: Split)

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


# def fine_tune_on_analogy(agent, vocab, analogy_data, epochs, batch_size, device):
#     """
#     Fine-tune the PPO agent on analogy tasks.
#     Args:
#         agent (PPOAgent): The agent to fine-tune.
#         vocab (Vocabulary): The vocabulary.
#         analogy_data (list): List of analogy pairs (tokenized).
#         epochs (int): Number of epochs to train.
#         batch_size (int): Batch size for training.
#         device (torch.device): Device to train on.
#     """

#     analogy_loader = DataLoader(analogy_data, batch_size=batch_size, shuffle=True)

#     for epoch in range(epochs):
#         total_loss = 0.0
#         for batch in analogy_loader:
#             word1, word2, word3, word4 = batch.split()  # Tokenized analogy words

#             # Generate embeddings for the analogy words
#             emb1 = agent.policy_network.embedding(word1.to(device))
#             emb2 = agent.policy_network.embedding(word2.to(device))
#             emb3 = agent.policy_network.embedding(word3.to(device))
#             emb4 = agent.policy_network.embedding(word4.to(device))

#             # Compute analogy loss: (emb2 - emb1) + emb3 ~= emb4
#             predicted_emb4 = emb2 - emb1 + emb3
#             loss = F.mse_loss(predicted_emb4, emb4)

#             # Backpropagation
#             agent.optimizer.zero_grad()
#             loss.backward()
#             agent.optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


# best_runs = []
#
# def objective(trial):
#     global best_runs
#
#     # Suggest hyperparameters
#     embedding_dim = trial.suggest_categorical("embedding_dim", [512, 4096])
#     learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-7)
#     context_size = trial.suggest_int("context_size", 100, 150)
#     clip_vale = trial.suggest_float("clip_value", 0.1, 0.2)
#
#     # New hyperparameters for rewards
#     length_bonus_weight = trial.suggest_float("length_bonus_weight", 7, 100)
#     frequency_bonus_weight = trial.suggest_float("frequency_bonus_weight", 7, 100)
#     pair_frequency_bonus_weight = trial.suggest_float("pair_frequency_bonus_weight", 4, 100)
#     short_token_penalty = trial.suggest_float("short_token_penalty", -100.0, -7)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     print("\n[Main] Setting up project directories.")
#     create_directory_structure()
#
#     print(f"\n[Main] Preparing '{DATASET_NAME}' dataset.")
#     download_and_preprocess_wikitext(dataset_name=DATASET_NAME, output_dir=DATA_PATH)
#
#     train_data = load_preprocessed_data(split=TRAIN_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
#     train_data = train_data[:int(len(train_data) * DATA_PERCENTAGE)]
#
#     validation_data = load_preprocessed_data(split=VALIDATION_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
#     validation_data = validation_data[:int(len(validation_data) * DATA_PERCENTAGE)]
#
#     vocab = Vocabulary(train_data)
#     data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR, drop_last=True, persistent_workers=True)
#
#     env_fns = [lambda: TokenizerEnvironment(
#         vocab=vocab,
#         context_size=context_size,
#         length_bonus_weight=length_bonus_weight,
#         frequency_bonus_weight=frequency_bonus_weight,
#         pair_frequency_bonus_weight=pair_frequency_bonus_weight,
#         short_token_penalty=short_token_penalty
#     ) for _ in range(BATCH_SIZE)]
#     envs = VectorizedTokenizerEnvironment(env_fns)
#
#     policy_network = TransformerPolicyNetwork(
#         vocab_size=vocab.size,
#         embedding_dim=embedding_dim,
#         num_heads=4,
#         num_layers=3,
#         action_size=2,
#         dropout=0.1
#     ).to(device)
#
#     agent = PPOAgent(
#         policy_network=policy_network,
#         lr=learning_rate,
#         gamma=0.99,
#         eps_clip=clip_vale,
#         k_epochs=3,
#         device=device
#     )
#
#     print("\n[Main] Starting training process.")
#     for epoch in range(1):  # Modify as needed for more epochs
#         train(agent, envs, vocab, data_loader, validation_data, epochs=1, device=device)
#
#         # Evaluate performance
#         accuracy = evaluate_word_analogies(
#             embeddings=policy_network.embedding.weight.detach().cpu(),
#             vocab=vocab,
#             analogy_file_path="data/questions-words.txt",
#             device=device
#         )
#
#         # Record current trial results
#         trial_result = {
#             "embedding_dim": embedding_dim,
#             "learning_rate": learning_rate,
#             "context_size": context_size,
#             "length_bonus_weight": length_bonus_weight,
#             "frequency_bonus_weight": frequency_bonus_weight,
#             "pair_frequency_bonus_weight": pair_frequency_bonus_weight,
#             "short_token_penalty": short_token_penalty,
#             "accuracy": accuracy
#         }
#
#         best_runs.append(trial_result)
#         best_runs = sorted(best_runs, key=lambda x: x["accuracy"], reverse=True)[:10]
#
#         # Print the top 5 best runs
#         print("\n[Main] Top 5 Training Runs:")
#         for i, run in enumerate(best_runs):
#             print(f"Rank {i + 1}: Accuracy = {run['accuracy']:.4f}, Params = {run}")
#
#     return accuracy



def main_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = [
        DATA_PATH, TRAIN_SPLIT, VALIDATION_SPLIT, DATASET_NAME, DATA_PERCENTAGE,
        STEPS_PER_EPISODE,
        BATCH_SIZE, NUM_ENVIRONMENTS, NUM_WORKERS, PREFETCH_FACTOR, EPOCHS,
        EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, ACTION_SIZE, LEARNING_RATE, GAMMA, EPS_CLIP, K_EPOCHS,
        CONTEXT_SIZE,
        MODEL_SAVE_PATH, MODEL_CHECKPOINT_INTERVAL,
        device
    ]

    # Print Parameters
    print_parameters(params)

    # Create necessary directories
    print("\n[Main] Setting up project directories.")
    create_directory_structure()

    # Download and preprocess dataset if not already done
    # print(f"\n[Main] Preparing '{DATASET_NAME}' dataset.")
    # download_and_preprocess_wikitext(dataset_name=DATASET_NAME, output_dir=DATA_PATH)
    print(f"\n[Main] Preparing gigaword dataset.")
    download_and_preprocess_giga(DATA_PATH)

    # Load training and validation data
    print("\n[Main] Loading training data.")
    # train_data = load_preprocessed_data(split=TRAIN_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
    train_data = load_preprocessed_giga(split=TRAIN_SPLIT, output_dir=DATA_PATH)
    train_data = train_data[:int(len(train_data) * DATA_PERCENTAGE)]

    print("[Main] Loading validation data.")
    # validation_data = load_preprocessed_data(split=VALIDATION_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
    validation_data = load_preprocessed_giga(split=VALIDATION_SPLIT, output_dir=DATA_PATH)
    validation_data = validation_data[:int(len(validation_data) * DATA_PERCENTAGE)]
    print(f"[Main] Loaded {len(train_data)} training examples.")
    print(f"[Main] Loaded {len(validation_data)} validation examples.")

    print("\n[Main] Initializing vocabulary...")
    vocab = Vocabulary(train_data)
    print(f"[Main] Vocabulary Size: {vocab.size}")

    print(f"\n[Main] Preparing DataLoader with batch size: {BATCH_SIZE}")
    data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR, drop_last=True)
    print("[Main] DataLoader prepared.")

    # Create vectorized environments
    print(f"\n[Main] Creating {NUM_ENVIRONMENTS} vectorized environments.")
    env_fns = [lambda: TokenizerEnvironment(vocab=vocab, context_size=CONTEXT_SIZE) for _ in range(NUM_ENVIRONMENTS)]
    envs = VectorizedTokenizerEnvironment(env_fns)
    print("[Main] Vectorized environments created.")

    # Initialize policy network
    print("\n[Main] Initializing policy network.")
    policy_network = TransformerPolicyNetwork(vocab_size=vocab.size, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, action_size=ACTION_SIZE, dropout=DROPOUT).to(device)
    print("[Main] Policy network initialized.")

    # Initialize PPO agent
    print("\n[Main] Initializing PPO agent.")
    agent = PPOAgent(policy_network=policy_network, lr=LEARNING_RATE, gamma=GAMMA, eps_clip=EPS_CLIP, k_epochs=K_EPOCHS, device=device)
    print("[Main] PPO agent initialized.")

    # Compute efficiency score
    efficiency_percentage, model_params, dataset_size, gpu_flops_tflops, compute_bottleneck, suggestion = compute_efficiency_score(policy_network, len(train_data))
    print(f"[Main] Model Parameters: {model_params}")
    print(f"[Main] Dataset Size: {dataset_size}")
    print(f"[Main] GPU FLOPs: {gpu_flops_tflops:.2f} TFLOPs")
    print(f"[Main] Compute Bottleneck: {compute_bottleneck}\n")

    print(f"[Main] Efficiency Score: {efficiency_percentage:.2f}%")
    print(f"[Main] Suggestion: {suggestion}")

    # Start training
    print("\n[Main] Starting training process.")
    sleep(0.1)  # Sleep to allow printing to complete in order
    train(agent, envs, vocab, data_loader, validation_data, epochs=EPOCHS, device=device)

def main_testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = [
    DATA_PATH, TRAIN_SPLIT, VALIDATION_SPLIT, DATASET_NAME, DATA_PERCENTAGE,
    STEPS_PER_EPISODE,
    BATCH_SIZE, NUM_ENVIRONMENTS, NUM_WORKERS, PREFETCH_FACTOR, EPOCHS,
    EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, ACTION_SIZE, LEARNING_RATE, GAMMA, EPS_CLIP, K_EPOCHS,
    CONTEXT_SIZE,
    MODEL_SAVE_PATH, MODEL_CHECKPOINT_INTERVAL,
    device
    ]

    # Print Parameters
    print_parameters(params)

    # Create necessary directories
    print("\n[Main] Setting up project directories.")
    create_directory_structure()

    # Download and preprocess dataset if not already done
    # print(f"\n[Main] Preparing '{DATASET_NAME}' dataset.")
    # download_and_preprocess_wikitext(dataset_name=DATASET_NAME, output_dir=DATA_PATH)
    print(f"\n[Main] Preparing gigaword dataset.")
    download_and_preprocess_giga(DATA_PATH)

    # Load training and validation data
    print("\n[Main] Loading training data.")
    # train_data = load_preprocessed_data(split=TRAIN_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
    train_data = load_preprocessed_giga(split=TRAIN_SPLIT, output_dir=DATA_PATH)
    train_data = train_data[:int(len(train_data) * DATA_PERCENTAGE)]

    print("[Main] Loading validation data.")
    # validation_data = load_preprocessed_data(split=VALIDATION_SPLIT, output_dir=DATA_PATH, dataset_name=DATASET_NAME)
    validation_data = load_preprocessed_giga(split=VALIDATION_SPLIT, output_dir=DATA_PATH)
    validation_data = validation_data[:int(len(validation_data) * DATA_PERCENTAGE)]
    print(f"[Main] Loaded {len(train_data)} training examples.")
    print(f"[Main] Loaded {len(validation_data)} validation examples.")

    print("\n[Main] Initializing vocabulary...")
    vocab = Vocabulary(train_data)
    print(f"[Main] Vocabulary Size: {vocab.size}")

    print("\n[Main] Initializing policy network.")
    policy_network = TransformerPolicyNetwork(vocab_size=vocab.size, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, action_size=ACTION_SIZE, dropout=DROPOUT).to(device)
    print("[Main] Policy network initialized.")

    cbow_model = train_cbow(train_data, EMBEDDING_DIM)
    cbow_embeddings = get_word2vec_embeddings(cbow_model, vocab)
    cbow_accuracy = evaluate_word_analogies(cbow_embeddings, vocab, "data/questions-words.txt", device=device)
    print(f"[Main] Finished testing. Accuracy on word analogies for cbow: {cbow_accuracy*100:.2f}%")

    sleep(1)

    sg_model = train_skipgram(train_data, EMBEDDING_DIM)
    sg_embeddings = get_word2vec_embeddings(sg_model, vocab)
    sg_accuracy = evaluate_word_analogies(sg_embeddings, vocab, "data/questions-words.txt", device=device)
    print(f"[Main] Finished testing. Accuracy on word analogies for cbow: {sg_accuracy*100:.2f}%")

    sleep(1)

    print("\n[Main] Loading model.")
    policy_network.load_state_dict(torch.load(MODEL_SAVE_PATH))
    policy_network.eval()
    print("\n[Main] Model loaded.")

    print(f"[Main] Starting testing on analogies.")
    embeddings = policy_network.embedding.weight.detach().cpu()
    RL_accuracy = evaluate_word_analogies(embeddings, vocab, "data/questions-words.txt", device=device)
    print(f"[Main] Finished testing. Accuracy on word analogies: {RL_accuracy*100:.2f}%")

    print(f"Comparison of Cbow vs Skip Gram vs RL tokenizer on analogy: {cbow_accuracy*100:.2f}% vs {sg_accuracy*100:.2f}% vs {RL_accuracy*100:.2f}%")




# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective)
#
#     print("Best hyperparameters:", study.best_params)
#     print("Best accuracy:", study.best_value)

# if __name__ == "__main__":
#     main_training()

if __name__ == "__main__":
    main_testing()
