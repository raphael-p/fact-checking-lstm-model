from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Bidirectional
from keras.layers import Reshape
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import json
import sys
sys.path.append('../../lib')
from common_functions import *


def recall(y_true, y_pred):
    """
    custom recall metric for keras
    :param y_true: label data
    :param y_pred: neural network output
    :return: recall score for this network
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """
    custom precision metric for keras
    :param y_true: label data
    :param y_pred: neural network output
    :return: precision score for this network
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec


def f1(y_true, y_pred):
    """
    custom F1 metric for keras
    :param y_true: label data
    :param y_pred: neural network output
    :return: F1 score for this network
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def docIndexer(doc_data, max_length, embedding_dict, embedding_dimension):
    """
    Generates indices for each unique token in a document, along with an embedding matrix
    based on GloVe embeddings.
    :param doc_data: collection of tokenised texts
    :param max_length: maximum length of document
    :param embedding_dict: dictionary matching token to GloVe embedding
    :param embedding_dimension: dimensionality of GloVe embeddings
    :return: indexed version of doc_data, size of doc_data vocabulary, matrix of embeddings based on index
    """
    # generate indexed document
    t = Tokenizer()
    t.fit_on_texts(doc_data)
    sequences = t.texts_to_sequences(doc_data)
    word_index = t.word_index
    vocab_size = len(word_index) + 1
    data = pad_sequences(sequences, maxlen=max_length, padding='post')

    # generate embedding matrix
    num_words = min(vocab_size, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, embedding_dimension))
    for word, i in word_index.items():
        if i > vocab_size:
            continue
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return data, vocab_size, embedding_matrix


def dataSplit(claim, sent, label_data, split):
    """
    split the data into a training set and a validation set
    :param claim: claim data
    :param sent: evidence data
    :param label_data: labels
    :param split: percentage of dataset to be dedicated to testing
    :return: all training and testing inputs for the network
    """
    size = claim.shape[0]
    perm = np.random.permutation(size)
    claim = claim[perm]
    sent = sent[perm]
    label_data = label_data[perm]
    test_num = int(split * size)

    claim_train_ = claim[:-test_num]
    evi_train_ = sent[:-test_num]
    y_train_ = label_data[:-test_num]
    claim_test_ = claim[-test_num:]
    evi_test_ = sent[-test_num:]
    y_test_ = label_data[-test_num:]
    return claim_train_, evi_train_, y_train_, claim_test_, evi_test_, y_test_


if __name__ == "__main__":
    # loading training and testing data
    with open("../data/claims.jsonl", 'r') as openfile:
        claims = json.loads(openfile.readlines()[0])
    with open("../data/evis.jsonl", 'r') as openfile:
        sentences = json.loads(openfile.readlines()[0])
    with open("../data/labels.jsonl", 'r') as openfile:
        labels = json.loads(openfile.readlines()[0])
    labels = np.array(labels).reshape(-1, 1)
    print("Data import complete.")

    # model parameters
    emb_dim = 50
    n_classes = 1
    n_neurons = int(emb_dim / 2)
    minibatch_size = 30
    epochs = 10
    test_split = 0.2
    claim_tokens = 50
    evi_tokens = 200

    # load glove embeddings
    embedding_dictionary = loadEmbeddings('../../data/glove.6B.' + str(emb_dim) + 'd.txt')  # load embedding dictionary

    # prepare claim and sentence data, split into training and testing sets
    claim_data, claim_vocab_size, claim_emb_matrix = docIndexer(claims, claim_tokens, embedding_dictionary, emb_dim)
    evi_data, evi_vocab_size, evi_emb_matrix = docIndexer(sentences, evi_tokens, embedding_dictionary, emb_dim)
    claim_train, evi_train, y_train, claim_test, evi_test, y_test = dataSplit(claim_data, evi_data, labels, test_split)

    # generate biLSTM evidence representation from pre-trained embeddings, with attention
    evi_input = Input(shape=(evi_tokens,), name='evidence_1')
    evi_embedding = Embedding(evi_vocab_size, emb_dim, weights=[evi_emb_matrix], input_length=evi_tokens,
                              trainable=False, mask_zero=True, name='evidence_2')(evi_input)
    evi_bilstm = Bidirectional(LSTM(n_neurons, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                               name="evidence_3_encoding")(evi_embedding)
    evi_att = Dense(n_neurons * 2, activation='tanh', name="evidence_4_attention")(evi_bilstm)
    evi_rep = Multiply(name="evidence_5")([evi_bilstm, evi_att])
    evi_rep = Lambda(lambda xin: K.sum(xin, axis=-2),
                     name="evidence_6_sum")(evi_rep)
    evi_rep = Reshape((1, n_neurons * 2), name="evidence_7")(evi_rep)

    # generate biLSTM claim representation from pre-trained embeddings, with attention
    claim_input = Input(shape=(claim_tokens,), name='claim_1')
    claim_embedding = Embedding(claim_vocab_size, emb_dim, weights=[claim_emb_matrix], input_length=claim_tokens,
                                trainable=False, mask_zero=True, name='claim_2')(claim_input)
    claim_bilstm = Bidirectional(LSTM(n_neurons, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                                 name="claim_3_encoding")(claim_embedding)
    claim_att = Dense(n_neurons * 2, activation='tanh', name="claim_4_attention")(claim_bilstm)
    claim_rep = Multiply(name="claim_5")([claim_bilstm, claim_att])
    claim_rep = Lambda(lambda xin: K.sum(xin, axis=-2),
                       name="claim_6_sum")(claim_rep)
    claim_rep = Reshape((1, n_neurons * 2), name="claim_7")(claim_rep)

    # generate biLSTM claim-evidence representation from trained representations, with attention
    claim_evi_concat = Concatenate(axis=-2, name="claim_evi_1")([claim_rep, evi_rep])
    claim_evi_bilstm = Bidirectional(LSTM(n_neurons, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                                     name="claim_evi_2_encoding")(claim_evi_concat)
    claim_evi_att = Dense(n_neurons * 2, activation='tanh', name="claim_evi_3_attention")(claim_evi_bilstm)
    claim_evi_rep = Multiply(name="claim_evi_4")([claim_evi_bilstm, claim_evi_att])
    claim_evi_rep = Lambda(lambda xin: K.sum(xin, axis=-2),
                           name="claim_evi_5_sum")(claim_evi_rep)
    out = Dense(n_classes, activation='sigmoid', name="out")(claim_evi_rep)

    # generate model
    model = Model(inputs=[claim_input, evi_input], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy', f1, precision, recall])

    # visualise model and monitor training progress
    plot_model(model, to_file="../data/model.png")
    print(model.summary())
    print("Minibatch size: " + str(minibatch_size))
    model.fit([claim_train, evi_train], y_train, validation_data=([claim_test[:500], evi_test[:500]], y_test[:500]),
              epochs=epochs, batch_size=minibatch_size, verbose=2)

    # final evaluation of the model
    scores = model.evaluate([claim_test, evi_test], y_test, verbose=0)
    print("Loss: %.2f%%" % (scores[0] * 100))
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("F1 score: %.2f%%" % (scores[2] * 100))
    print("Precision: %.2f%%" % (scores[3] * 100))
    print("Recall: %.2f%%" % (scores[4] * 100))
