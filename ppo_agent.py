import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
# import torch_optimizer as optim # pip install torch-optimizer for Shampoo optimizer

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class PPOAgent:
    def __init__(self, policy_network, lr, gamma, eps_clip, k_epochs, device):
        self.device = device
        self.policy_network = policy_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        self.reset_buffers()

    # Reset buffers
    def reset_buffers(self):
        self.states = []
        self.attention_masks = []
        self.actions = []
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []

    # Select an action based on policy
    def select_action(self, state_tensor, attention_mask):
        """Select an action based on policy."""
        action_probs, *_ = self.policy_network(state_tensor, attention_mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action, action_logprob, dist.entropy()

    # Append to buffers
    def store_transitions(self, states, attention_masks, actions, log_probs, state_values, rewards, dones):
        self.states.append(states)
        self.attention_masks.append(attention_masks)
        self.actions.append(actions)
        self.log_probs.append(log_probs.detach())
        self.state_values.append(state_values.detach())
        self.rewards.extend(rewards)
        self.dones.extend(dones)

    def compute_returns(self, rewards, dones, gamma):
        returns = [] # Returns for each timestep
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done: # Reset the discounted reward if done
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward) # Discounted reward
            returns.insert(0, discounted_reward) # Insert at the beginning of the list for correct order
        return torch.tensor(returns, dtype=torch.float, device=self.device)

    # Update policy
    def update_policy(self):
        states, attention_masks = self._pad_sequences() # Pad sequences
        actions, old_log_probs, state_values, returns, advantages = self._prepare_tensors() # Prepare tensors
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # Normalize advantages

        dataset = TensorDataset(states, attention_masks, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True) # mini-batch dataset

        self._process_batches(loader)

        self.reset_buffers()

    # Pad sequences to match the maximum sequence length
    def _pad_sequences(self):
        all_lengths = [state.size(1) for state in self.states]
        max_seq_length = max(all_lengths)

        padding_index = 0
        padded_states = []
        padded_attention_masks = []

        for state, attention_mask in zip(self.states, self.attention_masks): # iterate over states and attention masks
            seq_length = state.size(1) # Get the sequence length
            if seq_length < max_seq_length: # Check if padding is required
                pad_length = max_seq_length - seq_length
                pad_state = torch.full((state.size(0), pad_length), padding_index, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad_state], dim=1)

                pad_mask = torch.zeros((attention_mask.size(0), pad_length), dtype=attention_mask.dtype, device=attention_mask.device) # Create padding mask
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1) # Concatenate padding mask

            padded_states.append(state)
            padded_attention_masks.append(attention_mask)

        states = torch.cat(padded_states, dim=0)
        attention_masks = torch.cat(padded_attention_masks, dim=0)
        return states, attention_masks  # (batch_size, seq_length), (batch_size, seq_length)

    def _prepare_tensors(self):
        actions = torch.cat(self.actions, dim=0).to(self.device)
        old_log_probs = torch.cat(self.log_probs, dim=0).to(self.device)
        state_values = torch.cat(self.state_values, dim=0).to(self.device)
        returns = self.compute_returns(self.rewards, self.dones, self.gamma)
        advantages = returns - state_values
        return actions, old_log_probs, state_values, returns, advantages

    def _process_batches(self, loader):
        grad_accum_steps = 4 # Gradient accumulation steps
        self.optimizer.zero_grad()

        for epoch in range(self.K_epochs): # K_epochs for updating policy of PP0
            cumulative_loss = 0.0
            batch_loss_sum = 0.0
            batch_count = 0

            for batch_idx, (batch_states, batch_attention_masks, batch_actions, batch_old_log_probs, batch_returns, batch_advantages) in enumerate(loader):
                loss, policy_loss, value_loss, entropy = self._compute_losses(batch_states, batch_attention_masks, batch_actions, batch_old_log_probs, batch_returns, batch_advantages) # Compute losses

                (loss / grad_accum_steps).backward()
                batch_loss_sum += loss.item()
                batch_count += 1

                # Update the model every grad_accum_steps
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=2.0) # Clip gradients

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    cumulative_loss += batch_loss_sum
                    batch_loss_sum = 0.0  # Reset the batch loss accumulator

            print(f"[PPOAgent] K_Epoch {epoch + 1}/{self.K_epochs}, Loss: {cumulative_loss:.4f}")

    def _compute_losses(self, states, attention_masks, actions, old_log_probs, returns, advantages):
        old_log_probs, returns, advantages = old_log_probs.detach(), returns.detach(), advantages.detach()

        # Evaluate current policy (action probabilities and state values)
        action_probs, state_values = self.policy_network(states, attention_masks)

        # Distribution over actions
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()

        # New log probs
        new_log_probs = dist.log_prob(actions)

        # PPO Loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        loss_1 = ratios * advantages
        loss_2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        policy_loss = -torch.min(loss_1, loss_2).mean() + (-new_log_probs.mean())  # Combine NLL loss
        value_loss = F.mse_loss(state_values, returns)

        # Total loss (policy + value + entropy)
        loss = policy_loss + (0.5 * value_loss) - (0.03 * entropy)

        return loss, policy_loss, value_loss, entropy

    def save_model(self, path):
        torch.save(self.policy_network.state_dict(), path)
        print(f"[PPOAgent] Model saved to {path}")

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(state_dict)
        print(f"[PPOAgent] Model loaded from {path}")
