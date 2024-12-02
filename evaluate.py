import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader

from policy_network import TransformerPolicyNetwork
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from vocab import Vocabulary
from ppo_agent import PPOAgent
from tokenizer_environment import TokenizerEnvironment
from utils import preprocess_single_state, load_preprocessed_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = {
    "bpe_tokenizer": "tokenizers/bpe/bpe_tokenizer.json",
    "rl_policy_model": "models/final/final_policy_model.pth",
    "bert_save": "models/fine_tuned_bert",
    "data": "data"
}
dataset_name = "wikitext-2-raw-v1" # Options: 'wikitext-2-v1', 'wikitext-2-raw-v1', 'wikitext-103-v1', 'wikitext-103-raw-v1'

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"input_ids": tokens["input_ids"].squeeze(0), "attention_mask": tokens["attention_mask"].squeeze(0)}

def create_bpe_tokenizer(tokenizer_path, dataset):
    if not os.path.exists(tokenizer_path):
        bpe_tokenizer = Tokenizer(models.BPE())
        bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=55000, min_frequency=2, special_tokens=["[PAD]", "[UNK]"])
        bpe_tokenizer.train_from_iterator(dataset)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        bpe_tokenizer.save(tokenizer_path)
        print(f"[INFO] BPE tokenizer saved at {tokenizer_path}")
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))

def load_rl_policy(model_path, vocab_size):
    policy_network = TransformerPolicyNetwork(vocab_size=vocab_size, embedding_dim=128, num_heads=4, num_layers=5, action_size=2, dropout=0.1) # Make sure these align with the saved model
    rl_policy = PPOAgent(policy_network, lr=1e-6, gamma=0.99, eps_clip=0.2, k_epochs=3, device=device)
    rl_policy.load_model(model_path)
    return rl_policy

def rl_tokenize(text, vocab, rl_policy, max_length=256):
    env = TokenizerEnvironment(vocab=vocab, context_size=25) # Make sure these align with the saved model
    state = env.reset(text)
    tokens = []

    while not env.done:
        state_tensor, attention_mask = preprocess_single_state(state, vocab, max_length=max_length, device=device)
        action, *_ = rl_policy.select_action(state_tensor, attention_mask)
        state, _, env.done, _ = env.step(action.item())
        tokens.extend(vocab.encode(env.tokens))

    return tokens[:max_length]


def fine_tune_bert(model, data_loader, tokenizer, epochs=1, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        print(f"\n--- Fine-tuning Epoch {epoch + 1} ---")

        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()
            loss = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["input_ids"].to(device)).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")

        sample_input = batch["input_ids"][0].to(device)
        sample_output_ids = model.generate(input_ids=sample_input.unsqueeze(0), max_new_tokens=50)
        sample_output_text = tokenizer.decode(sample_output_ids[0], skip_special_tokens=True)
        print("Sample Output after Epoch", epoch + 1, ":", sample_output_text)


def evaluate_bert(model, tokenizer, tokenizer_type, data, vocab, sample_interval=0.1):
    model.eval()
    metrics = initialize_metrics_dict()
    total_freq = sum(vocab.token_frequency.values())

    sample_interval_count = int(len(data) * sample_interval)
    results = []

    with torch.no_grad():
        for idx, text in enumerate(tqdm(data, desc="Evaluating")):
            if tokenizer_type == "rl":
                tokens = rl_tokenize(text, vocab, rl_policy)
            else:
                tokens = bpe_tokenizer.encode(text, add_special_tokens=False)

            update_metrics(metrics, tokens, vocab, total_freq)

            if idx % sample_interval_count == 0:
                sample_output_ids = model.generate(input_ids=torch.tensor(tokens).unsqueeze(0).to(device), max_new_tokens=50)
                sample_output_text = tokenizer.decode(sample_output_ids[0], skip_special_tokens=True)
                results.append((idx, sample_output_text))
                print(f"Sample Output at {int((idx / len(data)) * 100)}% of Evaluation:", sample_output_text)

    final_metrics = calculate_final_metrics(metrics, len(data))
    print("Final Metrics:", final_metrics)

    return final_metrics, results


def initialize_metrics_dict():
    return {"total_tokens": 0, "total_token_length": 0, "total_perplexity": 0.0, "num_perplexity_evals": 0}

def update_metrics(metrics, tokens, vocab, total_freq):
    metrics["total_tokens"] += len(tokens)

    for token in tokens:
        token_length = len(str(token))
        metrics["total_token_length"] += token_length

    perplexity = compute_perplexity(tokens, vocab, total_freq)
    if perplexity:
        metrics["total_perplexity"] += perplexity
        metrics["num_perplexity_evals"] += 1


def compute_perplexity(tokens, vocab, total_freq):
    if not tokens:
        return None

    log_p_sum = 0
    valid_token_count = 0

    for token in tokens:
        frequency = vocab.get_frequency(token)
        if frequency > 0:
            log_p_sum += np.log(frequency / total_freq)
            valid_token_count += 1

    # If no valid tokens, return None or infinity
    if valid_token_count == 0:
        return float("inf")

    return np.exp(-log_p_sum / valid_token_count)


def calculate_final_metrics(metrics, num_samples):
    if num_samples > 0:
        avg_tokens_per_sample = metrics["total_tokens"] / num_samples
    else:
        avg_tokens_per_sample = 0.0

    if metrics["total_tokens"] > 0:
        avg_token_length = metrics["total_token_length"] / metrics["total_tokens"]
    else:
        avg_token_length = 0.0

    if metrics["num_perplexity_evals"] > 0:
        avg_perplexity = metrics["total_perplexity"] / metrics["num_perplexity_evals"]
    else:
        avg_perplexity = float("inf")

    return {"Avg Tokens per Sample": avg_tokens_per_sample, "Avg Token Length": avg_token_length, "Avg Perplexity": avg_perplexity}

train_data = load_preprocessed_data("train", paths["data"], dataset_name)
validation_data = load_preprocessed_data("validation", paths["data"], dataset_name)

bpe_tokenizer = create_bpe_tokenizer(paths["bpe_tokenizer"], train_data)
bpe_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
rl_vocab = Vocabulary(train_data)
rl_policy = load_rl_policy(paths["rl_policy_model"], rl_vocab.size)


bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
train_loader = DataLoader(TextDataset(train_data, bpe_tokenizer, max_length=256), batch_size=32, shuffle=True, pin_memory=True)


print("[Fine-tuning BERT with BPE Tokenizer]")
fine_tune_bert(bert_model, train_loader, bpe_tokenizer, epochs=5)


print("[Evaluating BERT with BPE Tokenizer]")
bpe_metrics, bpe_sample_outputs = evaluate_bert(
    model=bert_model,
    tokenizer=bpe_tokenizer,
    tokenizer_type="bpe",
    data=validation_data,
    vocab=rl_vocab  # Uses the RL vocab for comparison metrics
)

print("[Fine-tuning BERT with RL Tokenizer]")
fine_tune_bert(bert_model, train_loader, bpe_tokenizer, epochs=5)

print("[Evaluating BERT with RL Tokenizer]")
rl_metrics, rl_sample_outputs = evaluate_bert(
    model=bert_model,
    tokenizer=bpe_tokenizer,  # Decodes the output for readability
    tokenizer_type="rl",
    data=validation_data,
    vocab=rl_vocab
)

print("\n=== Evaluation Results ===")
print("BPE Tokenizer Metrics:", bpe_metrics)
print("RL Tokenizer Metrics:", rl_metrics)

# sample
print("\n=== Sample Outputs During Fine-Tuning ===")
for i, output in enumerate(rl_sample_outputs):
    print(f"RL Tokenizer Sample Output at {i*5}%:", output)

for i, output in enumerate(bpe_sample_outputs):
    print(f"BPE Tokenizer Sample Output at {i*5}%:", output)