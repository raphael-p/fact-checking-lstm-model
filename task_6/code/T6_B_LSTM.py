from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
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


if __name__ == "__main__":
    # loading training and testing data
    with open("../data/x_train.jsonl", 'r') as openfile:
        x_train = json.loads(openfile.readlines()[0])
    x_train = np.array(x_train)
    with open("../data/y_train.jsonl", 'r') as openfile:
        y_train = json.loads(openfile.readlines()[0])
    y_train = np.array(y_train).reshape(-1, 1)
    with open("../data/x_test.jsonl", 'r') as openfile:
        x_test = json.loads(openfile.readlines()[0])
    x_test = np.array(x_test)
    with open("../data/y_test.jsonl", 'r') as openfile:
        y_test = json.loads(openfile.readlines()[0])
    y_test = np.array(y_test).reshape(-1, 1)
    print("training input shape: "+str(x_train.shape))
    print("training label shape: "+str(y_train.shape))
    print("Dataset ready.")

    # define problem properties
    n_inputs = x_train.shape[0]
    timesteps = x_train.shape[1]
    input_size = x_train.shape[2]
    n_classes = 1
    n_neurons_1 = 300
    n_neurons_2 = 200
    minibatch_size = 30
    epochs = 15

    # define LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(n_neurons_1, return_sequences=False), input_shape=(timesteps, input_size)))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', f1, precision, recall])
    print(model.summary())
    print("Minibatch size: "+str(minibatch_size))
    print("Timesteps: "+str(timesteps))
    model.fit(x_train, y_train, validation_data=(x_test[:1000], y_test[:1000]),
              epochs=epochs, batch_size=minibatch_size, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Loss: %.2f%%" % (scores[0] * 100))
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("F1 score: %.2f%%" % (scores[2] * 100))
    print("Precision: %.2f%%" % (scores[3] * 100))
    print("Recall: %.2f%%" % (scores[4] * 100))
