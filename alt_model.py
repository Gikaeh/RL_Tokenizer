from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from tqdm import tqdm

def train_word2vec(training_data, embedding_dim):
    word2vec_model = Word2Vec(training_data, vector_size=embedding_dim, window=5, min_count=1, callbacks=[TqdmCallback(5)])
    return word2vec_model

def get_word2vec_embeddings(model, vocab):
    embedding_matrix = []
    for word in vocab.token_to_idx.keys():
        if word in model.wv:
            embedding_matrix.append(model.wv[word])
        else:
            embedding_matrix.append(np.zeros(model.vector_size))
    return np.array(embedding_matrix)

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