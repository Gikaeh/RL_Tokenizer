import numpy as np


def evaluate_word_analogies(embeddings, vocab, analogy_file_path, device):
    correct, total, skip = 0, 0, 0
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    with open(analogy_file_path, 'r') as f:
        for line in f:
            if line.startswith(':'):  # Skip category headers
                continue
            word_a, word_b, word_c, word_d = line.strip().split()
            
            # Converts all tokens to indexes to get the embeddings
            idx_a = vocab.token_to_idx.get(word_a.lower(), vocab.token_to_idx["<unk>"])
            idx_b = vocab.token_to_idx.get(word_b.lower(), vocab.token_to_idx["<unk>"])
            idx_c = vocab.token_to_idx.get(word_c.lower(), vocab.token_to_idx["<unk>"])
            idx_d = vocab.token_to_idx.get(word_d.lower(), vocab.token_to_idx["<unk>"])

            # Checks if any of the indexes are 1 (unknown) and adds bpe token to front of word
            if idx_a == 1:
                idx_a = vocab.token_to_idx.get(('Ġ' + word_a.lower()), vocab.token_to_idx["<unk>"])
            if idx_b == 1:
                idx_b = vocab.token_to_idx.get(('Ġ' + word_b.lower()), vocab.token_to_idx["<unk>"])
            if idx_c == 1:
                idx_c = vocab.token_to_idx.get(('Ġ' + word_c.lower()), vocab.token_to_idx["<unk>"])
            if idx_d == 1:
                idx_d = vocab.token_to_idx.get(('Ġ' + word_d.lower()), vocab.token_to_idx["<unk>"])

            # Checks if adding token to front of word works if not skip words
            if idx_a == 1 or idx_b == 1 or idx_c == 1 or idx_d == 1:
                skip += 1
                continue

            # Predicted vector: emb_d = emb_b - emb_a + emb_c
            predicted_emb = embeddings[idx_a] - embeddings[idx_b] + embeddings[idx_c]

            # Find the most similar word to the predicted embedding
            predicted_emb /= np.linalg.norm(predicted_emb)
            
            similarities = np.dot(embeddings, predicted_emb)
            similarities[idx_a, idx_b, idx_c] = -np.inf # Exclude original words from the nearest neighbor search
            best_match_idx = np.argmax(similarities)

            # Check if the prediction is correct
            if best_match_idx == idx_d:
                correct += 1
            total += 1    

            # if idx_a != 1:
            #     print('Word 1: ', vocab.idx_to_token.get(idx_a))
            print('Word 1: ', vocab.idx_to_token.get(idx_a), 'Word 2: ', vocab.idx_to_token.get(idx_b), 'Word 3: ', vocab.idx_to_token.get(idx_c), 'Word 4: ', vocab.idx_to_token.get(idx_d), 'predicted word: ', vocab.idx_to_token.get(best_match_idx))
            print(idx_a, ' ', idx_b, ' ', idx_c, ' ', idx_d, ' ', best_match_idx)

    return correct/total      