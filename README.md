# Tokenization with PPO-based Transformer Policy

## Project Overview

This project aims to develop an intelligent and context-aware tokenizer using reinforcement learning (RL) with Proximal Policy Optimization (PPO). Traditional tokenization techniques like Byte-Pair Encoding (BPE) rely on predefined rules, which can lack flexibility in diverse contexts. Our approach leverages a Transformer-based policy network to dynamically determine the optimal tokenization strategy during text processing. By training the tokenizer in an RL environment, the system can adapt to the complexities of language, balancing token size, frequency, and contextual meaning to produce more efficient tokenization schemes.

The tokenizer interacts with the TokenizerEnvironment, which simulates a dynamic text segmentation process. Actions (continue reading or split text) are chosen by the PPO agent based on a reward system that accounts for token length, frequency, and context, leading to a more intelligent and adaptive tokenization approach.

## Key Objectives

- Improve upon standard tokenization methods by using reinforcement learning.
- Train the model to dynamically determine when and how to split tokens in a way that is sensitive to both token frequency and contextual relevance.
- Evaluate the performance of the RL-based tokenizer against traditional methods (BPE) and models like GPT-2 and BERT.

## Key Features

### 1. Reinforcement Learning Tokenization

The tokenizer is trained using PPO, which allows it to learn tokenization strategies that are more adaptive compared to traditional methods. The RL environment rewards token splits that balance between contextual awareness, token frequency, and minimizing rare tokens.

- TokenizerEnvironment: A custom RL environment simulating tokenization as a decision-making process.
- PPO-based Optimization: The agent learns through trial and error using rewards that encourage optimal tokenization behavior.

### 2. Transformer-based Policy Network

A Transformer policy network is used to model the complex relationship between tokenization decisions and contextual information. The network processes text data through attention layers to extract relevant patterns before making tokenization decisions.

- Context-aware Tokenization: The model considers not just the current token but the context surrounding it to make more informed decisions.
- Action and Value Estimation: The policy network predicts both the action (split or continue) and the value (expected reward) for each tokenization step.

### 3. Dynamic Reward System

The agent receives rewards based on several factors:

- Token Length Bonus: Rewards token splits that yield tokens of appropriate lengths.
- Frequency Bonus: Rewards common tokens more heavily to ensure that frequent words are prioritized.
- Penalty for Short Tokens: Discourages splitting into excessively short tokens, which can lead to inefficient tokenization.

### 4. Vectorized Environments

For efficient training, multiple tokenization environments are run in parallel, allowing the agent to learn from several tokenization tasks simultaneously.

### 5. Extensive Evaluation Metrics

The tokenizer's performance is evaluated using multiple metrics, including BLEU score, perplexity, and token coverage. The model is compared against traditional BPE tokenizers and base tokenizers from GPT-2 and BERT to assess its improvements.

- Perplexity: Measures the tokenization efficiency with language models like GPT-2 and BERT.
- BLEU Score: Assesses the overlap between generated tokens and the ground truth tokens.
- Token Coverage: Evaluates how well the tokenizer captures important tokens in a text.

## Technical Details

### 1. PPO Agent Architecture

The PPO agent is responsible for learning optimal tokenization strategies. It interacts with the TokenizerEnvironment, where it observes the current text and previous token, chooses actions (split or continue), and receives rewards. The PPO algorithm is used to update the policy network in a stable manner.

- Policy Network: A Transformer model with attention heads that analyze the context and predict actions.
- Action Space: Simple binary actions (0: Continue, 1: Split) to decide whether to keep processing text or split at the current position.
- Training: The agent stores experiences (states, actions, rewards) in buffers, computes advantages, and updates the policy network periodically using gradient-based optimization.

### 2. Transformer Policy Network

This network serves as the brain of the tokenization agent, processing tokenized inputs and outputting both action probabilities and value estimates for each tokenization step. The network uses multi-headed self-attention to capture dependencies between tokens and context.

- Rotary Position Encoding: Improves token representation by adding positional information into token embeddings.
- Action Head: Outputs probabilities for possible actions (split or continue).
- Value Head: Outputs value estimates, helping the agent assess the long-term reward for each action.

### 3. Tokenizer Environment

The environment models the tokenization process as an RL problem. Each step simulates reading through the text, deciding when to split tokens based on the current state (context and position in the text). The environment calculates rewards and checks if the text has been fully processed.

- Context Window: The environment provides a window of text around the current token to help the agent make context-sensitive decisions.
- Dynamic Token Rewards: Tokens are evaluated for their frequency, length, and how well they match patterns of common language usage.

### 4. Vocabulary and Token Management

The vocabulary is dynamically built and managed through the Vocabulary class, which also supports token frequency tracking. The vocabulary plays a key role in calculating the reward for token frequency, as well as encoding and decoding tokens during the tokenization process.

- Byte-Pair Encoding (BPE): The project also integrates a traditional BPE tokenizer to compare with the RL-based tokenizer.

### 5. Evaluation with Language Models

Post-training, the tokenizer is evaluated by passing its tokenized outputs through models like GPT-2 and BERT to calculate performance metrics such as perplexity and BLEU score.

- GPT-2/BERT Models: Pre-trained language models are used for computing metrics to validate the quality of the generated tokenizations.
- Baseline Comparisons: The RL-based tokenizer is compared against standard tokenizers used in GPT-2 and BERT to assess improvements.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch 1.9+
- HuggingFace Transformers
- TQDM, Numpy, NLTK

## Future Work

- Extend the action space to allow more complex tokenization decisions.
- Incorporate additional language models for evaluation.
- Experiment with different reward functions to enhance tokenizer adaptability across languages.