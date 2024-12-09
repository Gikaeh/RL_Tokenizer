import numpy as np
import torch
from scipy.stats import spearmanr

def evaluate_word_analogies(embeddings, vocab, analogy_file_path, model='RL'):
    correct, total, skip = 0, 0, 0

    # Convert numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError("[Error] Invalid embeddings before normalization.")

    zero_rows = np.linalg.norm(embeddings, axis=1) == 0
    if np.any(zero_rows):
        # print(f"[Warning] Replacing {np.sum(zero_rows)} zero embeddings with random values.")
        embeddings[zero_rows] = np.random.normal(scale=1e-6, size=(np.sum(zero_rows), embeddings.shape[1]))

    # Normalize embeddings
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError("[Error] Invalid embeddings after normalization.")

    with open(analogy_file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line.startswith(':'):
                continue

            word_a, word_b, word_c, word_d = line.strip().split() # Read analogy words.

            # print(f"[Debug] Processing analogy line {line_num}: {word_a}, {word_b}, {word_c}, {word_d}")

            if model == 'RL':
                # Map words to indices
                idx_a = vocab.token_to_idx.get(word_a.lower(), vocab.token_to_idx["<unk>"])
                idx_b = vocab.token_to_idx.get(word_b.lower(), vocab.token_to_idx["<unk>"])
                idx_c = vocab.token_to_idx.get(word_c.lower(), vocab.token_to_idx["<unk>"])
                idx_d = vocab.token_to_idx.get(word_d.lower(), vocab.token_to_idx["<unk>"])

                # Handle `Ġ`
                if idx_a == vocab.token_to_idx["<unk>"]:
                    idx_a = vocab.token_to_idx.get('Ġ' + word_a.lower(), vocab.token_to_idx["<unk>"])
                if idx_b == vocab.token_to_idx["<unk>"]:
                    idx_b = vocab.token_to_idx.get('Ġ' + word_b.lower(), vocab.token_to_idx["<unk>"])
                if idx_c == vocab.token_to_idx["<unk>"]:
                    idx_c = vocab.token_to_idx.get('Ġ' + word_c.lower(), vocab.token_to_idx["<unk>"])
                if idx_d == vocab.token_to_idx["<unk>"]:
                    idx_d = vocab.token_to_idx.get('Ġ' + word_d.lower(), vocab.token_to_idx["<unk>"])

                # Check if words in the vocab.
                # print(f"[Debug] Vocabulary indices: {word_a} -> {idx_a}, {word_b} -> {idx_b}, "
                #     f"{word_c} -> {idx_c}, {word_d} -> {idx_d}")

                # Skip unknown.
                if idx_a == vocab.token_to_idx["<unk>"] or idx_b == vocab.token_to_idx["<unk>"] or \
                idx_c == vocab.token_to_idx["<unk>"] or idx_d == vocab.token_to_idx["<unk>"]:
                    # print(f"[Warning] Skipping line {line_num}, <unk>.")
                    skip += 1
                    continue
            else:
                idx_a = vocab.get(word_a.lower(), None)
                idx_b = vocab.get(word_b.lower(), None)
                idx_c = vocab.get(word_c.lower(), None)
                idx_d = vocab.get(word_d.lower(), None)
                
                if idx_a == None or idx_b == None or idx_c == None or idx_d == None:
                    skip += 1
                    continue

            # predicted embedding = emb(b) - emb(a) + emb(c)
            predicted_emb = embeddings[idx_b] - embeddings[idx_a] + embeddings[idx_c]
            predicted_emb /= np.linalg.norm(predicted_emb)  # Normalize predicted embedding.

            if np.any(np.isnan(predicted_emb)) or np.any(np.isinf(predicted_emb)):
                # print(f"[Error] Invalid prediction. Skipping analogy.")
                skip += 1
                continue

            # Compute similarities
            similarities = np.dot(embeddings, predicted_emb)

            similarities[[idx_a, idx_b, idx_c]] = -np.inf

            best_match_idx = np.argmax(similarities) # Find the best match.

            top_words = []
            top_similarities = []
            top_indices = np.argsort(similarities)[-5:][::-1]

            # get top-5 predictions
            # for idx in top_indices:
            #     word = vocab.idx_to_token.get(idx, "<unk>")
            #     top_words.append(word)
            #     similarity = similarities[idx]
            #     top_similarities.append(similarity)

            # print(f"[Debug] Top Predictions: {list(zip(top_words, top_similarities))}")

            # Remove `Ġ`
            # predicted_word = vocab.idx_to_token.get(best_match_idx, "").lstrip('Ġ')
            # correct_word = vocab.idx_to_token.get(idx_d, "").lstrip('Ġ')

            if best_match_idx == idx_d:
                correct += 1
            total += 1

            # print(f"[Debug] Predicted: {predicted_word}, Correct: {correct_word}")

    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0

    # Debug: Final summary of evaluation.
    print(f"[Summary] Evaluation complete. Total analogies: {total + skip}, "
          f"Evaluated: {total}, Skipped: {skip}, Accuracy: {accuracy*100:.2f}")

    return accuracy


def evaluate_word_similarity(embeddings, vocab, simlex_path, model='RL'):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    zero_rows = np.linalg.norm(embeddings, axis=1) == 0
    # invalid_rows = np.isnan(embeddings).any(axis=1) | np.isinf(embeddings).any(axis=1)

    if np.any(zero_rows):
        print(f"[Warning] {np.sum(zero_rows)} zero embeddings found. Replacing with random values.")
        embeddings[zero_rows] = np.random.normal(scale=1e-6, size=(np.sum(zero_rows), embeddings.shape[1]))

    # if np.any(invalid_rows):
    #     print(f"[Warning] {np.sum(invalid_rows)} invalid embeddings found. Replacing with random values.")
    #     embeddings[invalid_rows] = np.random.normal(scale=1e-6, size=(np.sum(invalid_rows), embeddings.shape[1]))

    # Normalize embeddings
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError("[Error] Invalid embeddings after normalization.")
    
    predicted_similarities = []
    human_similarities = []
    skipped_pairs = 0

    # Read SimLex-999 file
    with open(simlex_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line_num, line in enumerate(file, start=1):
            parts = line.strip().split('\t')
            if len(parts) < 4:
                print(f"[Warning] Skipping line {line_num}: insufficient columns.")
                continue
            
            word1, word2, pos, simlex_score = parts[:4]
            
            if model == 'RL':
                # Convert words to indices
                idx1 = vocab.token_to_idx.get(word1.lower(), vocab.token_to_idx["<unk>"])
                idx2 = vocab.token_to_idx.get(word2.lower(), vocab.token_to_idx["<unk>"])
                
                # Handle BPE prefixes (Ġ)
                if idx1 == vocab.token_to_idx["<unk>"]:
                    idx1 = vocab.token_to_idx.get('Ġ' + word1.lower(), vocab.token_to_idx["<unk>"])
                if idx2 == vocab.token_to_idx["<unk>"]:
                    idx2 = vocab.token_to_idx.get('Ġ' + word2.lower(), vocab.token_to_idx["<unk>"])
                
                if idx1 == vocab.token_to_idx["<unk>"] or idx2 == vocab.token_to_idx["<unk>"]:
                    skipped_pairs += 1
                    continue
            else:
                idx1 = vocab.get(word1.lower(), None)
                idx2 = vocab.get(word2.lower(), None)
                
                if idx1 == None or idx2 == None:
                    skipped_pairs += 1
                    continue

            # Get embeddings for both words
            emb1 = embeddings[idx1]
            emb2 = embeddings[idx2]
            
            # Calculate cosine similarity
            cosine_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)  # Clamp values to [-1, 1]
            
            # Append similarity scores
            predicted_similarities.append(cosine_similarity)
            human_similarities.append(float(simlex_score))
    
    # Calculate mean squared error (MSE)
    if len(predicted_similarities) == 0:
        raise ValueError("[Error] No valid word pairs found in the dataset.")
    
    predicted_similarities = np.array(predicted_similarities)
    human_similarities = np.array(human_similarities)

    spearman_corr, _ = spearmanr(predicted_similarities, human_similarities)
    print(f"[Summary] Spearman correlation: {spearman_corr:.4f}")
    
    return spearman_corr
