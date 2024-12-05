import json
import os
from collections import Counter

from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel

from tokenizers import Tokenizer, models, trainers, normalizers


class Vocabulary:
    def __init__(self, texts):
        # Initialize token_to_idx and idx_to_token
        self.token_to_idx = {'<pad>': 0, '<unk>': 1} # Map special tokens to indices
        self.idx_to_token = {0: '<pad>', 1: '<unk>'} # Map indices to special tokens
        self.token_frequency = Counter({'<pad>': 0, '<unk>': 0}) # Frequency Counter
        self.next_idx = len(self.token_to_idx)  # Track the next available index

        # Initialize Hugging Face BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>")) # Initialize BPE tokenizer
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()]) # Normalization
        self.tokenizer.pre_tokenizer = ByteLevel() # Byte-level tokenization

        if texts:
            self.build_vocab(texts)

    # Build vocabulary from text inputs
    def build_vocab(self, texts):
        # Vocab size is set to upperbound of monolingual data # https://arxiv.org/abs/2310.08754
        # Monolingual lower bound is 33k, upper bound is 55k
        # Multiligual needs at least 100k
        combined_text = (f"{item['document']} {item['summary']}" for item in texts)
        trainer = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>"], vocab_size=33230)
        self.tokenizer.train_from_iterator(combined_text, trainer) # Train tokenizer on text inputs

        # Get the tokenizer vocabulary
        vocab = self.tokenizer.get_vocab()

        # Updates
        self.token_to_idx.update(vocab)
        self.idx_to_token.update({idx: token for token, idx in vocab.items()})
        self.next_idx = len(self.token_to_idx)  # Update the next available index

        # Update token frequency based on vocab
        self.token_frequency.update(vocab.keys())

    # Tokenize text using BPE tokenizer
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    # Add token to vocabulary
    def add_token(self, token: str):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.next_idx
            self.idx_to_token[self.next_idx] = token
            self.token_frequency[token] += 1
            self.next_idx += 1

    def get_frequency(self, token):
        return self.token_frequency.get(token, 0)

    # Encode tokens to indices
    def encode(self, tokens):
        return [self.token_to_idx.get(token, self.token_to_idx["<unk>"]) for token in tokens]

    # Decode indices to tokens
    def decode(self, indices):
        return [self.idx_to_token.get(idx, "<unk>") for idx in indices]

    # Get vocabulary size
    @property
    def size(self):
        return len(self.token_to_idx)

    # Save vocabulary to file
    def save(self, file_path):
        data = {"token_to_idx": self.token_to_idx, "idx_to_token": self.idx_to_token, "token_frequency": self.token_frequency}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[Vocabulary] Vocabulary saved to {file_path}")

    # Load vocabulary from file
    def load(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = Vocabulary(texts=[])
        vocab.token_to_idx = data['token_to_idx']
        vocab.idx_to_token = {int(k): v for k, v in data['idx_to_token'].items()}
        vocab.token_frequency = Counter(data['token_frequency'])
        print(f"[Vocabulary] Vocabulary loaded from {file_path}")
        return vocab
