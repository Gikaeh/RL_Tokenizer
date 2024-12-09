from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from tqdm import tqdm
import torch
from nltk.tokenize import word_tokenize
import nltk

def train_cbow(training_data, embedding_dim):
    word2vec_model = Word2Vec(training_data, vector_size=embedding_dim, window=5, min_count=1, callbacks=[TqdmCallback(5)])
    return word2vec_model

def train_skipgram(training_data, embedding_dim):
    word2vec_model = Word2Vec(training_data, vector_size=embedding_dim, window=5, min_count=1, sg=1, callbacks=[TqdmCallback(5)])
    return word2vec_model

def get_word2vec_embeddings(model, vocab):
    # Initialize the embedding matrix
    embedding_matrix = []
    vector_size = model.vector_size

    for word, idx in vocab.items():  # `vocab` is now a dictionary
        if word in model.wv:
            embedding_matrix.append(model.wv[word])
        else:
            embedding_matrix.append(np.zeros(vector_size))  # Use zeros for OOV words
    
    return np.array(embedding_matrix)

def load_w2v_vocab(model):
    return {word: idx for idx, word in enumerate(model.wv.key_to_index.keys())}

def w2v_tokenizer(data):
    # nltk.download('punkt_tab')
    sentences = []
    entrys = []
    for i in range(len(data)):
        entrys.append({"text": data[i]})

    for entry in entrys:
        document_tokens = word_tokenize(entry["text"].lower())  # Tokenize document
        # summary_tokens = word_tokenize(entry["summary"].lower())    # Tokenize summary
        sentences.append(document_tokens)
        # sentences.append(summary_tokens)

    return sentences

class TqdmCallback(CallbackAny2Vec):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self, model):
        self.pbar = tqdm(total=self.total_epochs, desc="Training Word2Vec")

    def on_epoch_end(self, model):
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()