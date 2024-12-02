# Reinforcement Learning for Tokenization

## Overview
In this project, we apply reinforcement learning to the task of tokenization and explore these capabilities.

## Files
- `main.py`: Main file conducting training and evaluations.
- `vocab.py`: Handles vocabulary creation and tokenization management.
- `utils.py`: Utilities file.
- `tokenizer_environment.py`: Defines the environment for the RL model.
- `ppo_agent.py`: The agent for navigating through the environment.
- `policy_network.py`: Transformer encoder policy network with RoPE.
- `evaluate.py`: Provides early methods for evaluation and fine-tuning.
- `test.py`: Runnings embedding evaluations on the model.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

```bash
conda env create -f environment.yml
conda activate LLM_Project
```

## Running the Program
Running the program is a bit tricky.

To run the program, head to main.py and scroll down. There will be three main code functions: objective(trials), main_training(), main_testing(). 
objective(trials) conducts a hyperparameter sweep to identify good parameters, main_training() trainings the model on the provided hyperparameters, and main_testing() evaluates the embedding space of our model.
If you continue to scroll down there will also be three functions titled `if __name__ == "__main__":`.
To run any of these approaches you must first uncomment the large block function you want to run, with the associated `if __name__ == "__main__":` function, then comment out the other two functions and their main functions.

There may be instances of failure due to line # 50 in `evaluate.py` or more detailed `policy_network = TransformerPolicyNetwork(.....)`, the variables within this line in the `load_rl_policy()` function should match your training parameters.

That should be all it takes. 

We have also included a model in the github repo to use as for testing: embedding_dim = 256, num_heads=9, num_layers=5

Our hyperparameter sweep gave the following: (embedding_dim=512, num_heads=8, num_layers=7, learning_rate=9.0e-7, RoPE Frequency(In `policy_network.py`)=6500, Reward Bonuses(in `tokenizer_environment.py`):(LENGTH_BONUS: 7.4, FREQUENCY_BONUS: 9.1, PAIR_FREQUENCY: 4.4, SHORT_TOKEN: -5.4))