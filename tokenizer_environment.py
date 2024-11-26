import numpy as np
from numpy.f2py.crackfortran import previous_context


class TokenizerEnvironment:
    LENGTH_BONUS_WEIGHT = 1.00
    FREQUENCY_BONUS_WEIGHT = 1.20
    PAIR_FREQUENCY_BONUS_WEIGHT = 1.10
    SHORT_TOKEN_PENALTY = -0.70

    def __init__(self, vocab, context_size):
        self.context_size = context_size # Context Window for Tokens
        self.vocab = vocab # Vocabulary object
        self.text = "" # Current text to tokenize
        self.position = 0 # Current position in the text
        self.tokens = []
        self.done = False
        self.reward_scale = 1.0

    #  Reset the environment
    def reset(self, new_text):
        self.text = new_text
        self.position = 0
        self.tokens = []
        self.done = False
        state = self.get_state()
        return state

    def get_state(self):
        start = max(self.position - self.context_size, 0)
        end = min(self.position + self.context_size, len(self.text))
        current_context = self.text[start:end] # Extract context window
        if self.tokens:
            previous_token = self.tokens[-1] # Get the previous token
        else:
            previous_token = "" # No previous token
        state = {'current_context': current_context, 'previous_token': previous_token, 'position': self.position} # State dictionary
        return state

    def step(self, action):
        reward = 0.0

        if action == 0: # Action 0: Continue tokenization
            reward = self._handle_continue()
        elif action == 1: # Action 1: Split token
            reward = self._handle_split()
        else:
            reward = self.SHORT_TOKEN_PENALTY * self.reward_scale # Apply penalty for invalid action
            print(f"[TokenizerEnvironment] Invalid action. Applied penalty.")

        self._check_done()

        state = self.get_state()

        # If done, calculate final token length for padding else return 0
        if self.done:
            token_length = len(self.tokens[-1])
        else:
            token_length = 0

        return state, reward, self.done, token_length

    def _handle_continue(self):
        if self.position < len(self.text): # Check if valid
            self.position += 1
            reward = -0.01 * self.reward_scale
        else:
            reward = -0.04 * self.reward_scale # Higher penalty for invalid continuation
        return reward

    def _handle_split(self):
        if self.position > 0:
            token = self.text[:self.position]
            self.tokens.append(token)
            self.text = self.text[self.position:]
            self.position = 0

            length_bonus = self._calculate_length_bonus(token)
            frequency_bonus = self._calculate_frequency_bonus(token)
            pair_frequency_bonus = self._calculate_pair_frequency_bonus(token)
            short_token_penalty = self._calculate_short_token_penalty(token)

            length_reward = self.LENGTH_BONUS_WEIGHT * length_bonus
            frequency_reward = self.FREQUENCY_BONUS_WEIGHT * frequency_bonus
            pair_frequency_reward = self.PAIR_FREQUENCY_BONUS_WEIGHT * pair_frequency_bonus

            total_reward = length_reward + frequency_reward + pair_frequency_reward + short_token_penalty
            return total_reward
        else:
            reward = 0.0  # No token to split at position 0 (start of text)
            return reward

    # Check if tokenization is done
    def _check_done(self):
        if self.position >= len(self.text):
            if self.text:
                token = self.text
                self.tokens.append(token)
            self.done = True

    # Calculate length bonus for token
    def _calculate_length_bonus(self, token):
        length = len(token)
        if 3 <= length <= 10:
            bonus = (length - 2) * self.LENGTH_BONUS_WEIGHT * self.reward_scale
            return bonus
        return 0.0

    # Calculate penalty for short tokens
    def _calculate_short_token_penalty(self, token):
        length = len(token)
        if length < 3:
            penalty = self.SHORT_TOKEN_PENALTY * self.reward_scale
            return penalty
        return 0.0

    # Calculate frequency bonus for token
    def _calculate_frequency_bonus(self, token):
        frequency = self.vocab.get_frequency(token) # Get token frequency
        if frequency > 0: # Check if token is in vocab
            log_frequency = np.log(frequency + 1)
            bonus = log_frequency * self.FREQUENCY_BONUS_WEIGHT * self.reward_scale
            return bonus
        else:
            return -0.1 * self.reward_scale  # Penalize rare tokens

    # Calculate pair frequency bonus for token
    def _calculate_pair_frequency_bonus(self, token):
        if self.tokens: # Check if previous tokens exist
            previous_token = self.tokens[-1]
            pair = previous_token + token
            frequency = self.vocab.get_frequency(pair)
            if frequency > 0:
                log_frequency = np.log(frequency + 1)
                bonus = log_frequency * self.PAIR_FREQUENCY_BONUS_WEIGHT * self.reward_scale
                return bonus
        return 0.0 # No bonus if pair frequency is 0


# Highly Optimized Vectorized Tokenizer Environment for Batch Processing (Parallel Environments)
class VectorizedTokenizerEnvironment:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns] # Initialize environments
        self.num_envs = len(self.envs) # Number of environments

    # Reset the environments
    def reset(self, batch_texts):
        assert len(batch_texts) == self.num_envs, "Number of texts must match number of environments."
        states = []
        for idx, (env, text) in enumerate(zip(self.envs, batch_texts)):
            state = env.reset(text)
            states.append(state)
        return states

    # Step through the environments
    def step(self, actions):
        assert len(actions) == self.num_envs, "Number of actions must match number of environments."
        next_states = []
        rewards = []
        dones = []
        token_lengths = []

        for idx, (env, action) in enumerate(zip(self.envs, actions)): # Iterate over environments and actions
            state, reward, done, token_length = env.step(action) # Take step
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            token_lengths.append(token_length)

        return next_states, rewards, dones, token_lengths
