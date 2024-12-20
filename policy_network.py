import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, action_size, dropout):
        super(TransformerPolicyNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rotary_position_encoding = RotaryEmbedding(embedding_dim)  # Use rotary position encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)  # Stack layers
        self.action_head = nn.Linear(embedding_dim, action_size)  # Output layer for action
        self.value_head = nn.Linear(embedding_dim, 1)  # Output layer for state value

    def forward(self, x, attention_mask):
        embedded = self.rotary_position_encoding(self.embedding(x))
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=~attention_mask.bool())
        last_outputs = transformer_output.mean(dim=1)  # Average over sequence length

        logits = self.action_head(last_outputs)  # Action prediction
        action_probs = F.softmax(logits, dim=-1)
        state_values = self.value_head(last_outputs).squeeze(-1)

        return action_probs, state_values


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        inv_freq = 1.0 / (7500 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        sinusoid_inp = torch.einsum('i,j->ij', positions, self.inv_freq)
        sin_emb = sinusoid_inp.sin()[None, :, :].to(torch.float32)
        cos_emb = sinusoid_inp.cos()[None, :, :].to(torch.float32)
        return self.apply_rotary_embedding(x, sin_emb, cos_emb)

    def apply_rotary_embedding(self, x, sin_emb, cos_emb):
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        x_rot = torch.cat([x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1)  # Apply rotation
        return x_rot
