import sys
sys.path.append('../../lib/')
from common_functions import *


def positiveList(ids, training_file, results_file):
    """
    retrieve evidence sentences from train file
    :param ids: id names of documents for which we retrieve evidence
    :param training_file: file containing positive sentences for each document
    :return: list of lists for each evidence sentence
    format is [[claim_position, document_position, sentence_position],...]
    """
    doc_ids = docIds(directory=results_file)
    training_list = [[] for _ in range(10)]

    with open(training_file, 'r') as openfile:
        claim_pos = 0
        for iline, line in enumerate(openfile.readlines()):
            train_doc = json.loads(line)
            if train_doc["id"] in ids:
                for evidences in train_doc["evidence"]:
                    for evidence in evidences:
                        for doc_pos in range(5):
                            doc_id = doc_ids[claim_pos][doc_pos]
                            if doc_id == evidence[2]:
                                sent_pos = evidence[3]
                                sentence_arr = [claim_pos, doc_pos, sent_pos]
                                if sentence_arr in training_list[claim_pos]:
                                    continue
                                training_list[claim_pos].append(sentence_arr)
                claim_pos = claim_pos + 1
            if claim_pos > 9:
                return training_list
    return 0


def data(word_emb, pos_list):
    """
    prepares data for training and testing
    :param word_emb: word embeddings of each claim and corresponding results
    :param pos_list: list of relevant sentences for each claim
    :return: x and y data for either taining or testing
    """
    x = []
    y = np.array([])
    for i in range(len(word_emb)):
        claim_emb = np.array(word_emb[i][0])
        pos_sent = [pos_list[i][k][2] for k in range(len(pos_list[i]))]
        pos_doc = [pos_list[i][k][1] for k in range(len(pos_list[i]))]
        for doc_pos in range(5):
            N = len(word_emb[i][doc_pos+1])
            for sentence_pos in range(N):
                if (doc_pos in pos_doc) and (sentence_pos in pos_sent):
                    sentence_emb = np.array(word_emb[i][doc_pos+1][sentence_pos])
                    x_input = np.append(claim_emb, sentence_emb)
                    x.append(x_input)
                    y = np.append(y, 1)
                elif decision(0.7/N):
                    sentence_emb = np.array(word_emb[i][doc_pos+1][sentence_pos])
                    if not all(sentence_emb) == 0:
                        x_input = np.append(claim_emb, sentence_emb)
                        x.append(x_input)
                        y = np.append(y, 0)
    x = np.array(x)
    y = y.reshape(-1, 1)
    size = y.size
    perm = np.random.permutation(size)
    x = x[perm][:]
    y = y[perm]
    return x, y


def sigmoid(z):
    """
    sigmoid function, used in logistic regression
    :param z: function inputs
    :return: function outputs
    """
    return 1 / (1 + np.e**(-z))


def logisticLoss(y, y_hat):
    """
    logistic loss function
    :param y: expected result
    :param y_hat: model output
    :return: loss of model
    """
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def logisticRegression(xtrain, ytrain, xtest, ytest, epochs, learningrate, type=1):
    """
    trains and tests a logistic regression model
    :param xtrain: training inputs
    :param ytrain: training outputs
    :param xtest: testing inputs
    :param ytest: testing outputs
    :param epochs: number of training iterations
    :param learningrate: speed at which model learns
    :return: accuracy of test results
    """
    # defining shape of data
    weight_num = xtrain.shape[1]
    m = ytrain.size

    # initialise weights
    w = np.zeros((weight_num, 1))  # one weight per feature
    b = np.zeros((1, 1))  # one bias per data point

    # training
    for epoch in range(epochs):
        Z = np.matmul(xtrain, w) + b
        A = sigmoid(Z)
        loss = logisticLoss(ytrain, A)
        dJdZ = A - ytrain
        dJdw = (1 / m) * np.matmul(xtrain.T, dJdZ)
        dJdb = np.sum(dJdZ)
        w = w - learningrate * dJdw
        b = b - learningrate * dJdb

    # testing
    Z = np.matmul(xtest, w) + b
    test_preds = []
    recall = 0
    for i in sigmoid(Z):
        if i > 0.5:
            test_preds.append(1)
        if i < 0.5:
            test_preds.append(0)
    if type == 1:
        for j in range(len(test_preds)):
            if ytest[j] == 1 and test_preds[j] == 1:
                recall = recall + 1
        recall = recall / np.sum(ytest)
        accuracy = 1 - (np.sum(np.abs(ytest.ravel() - np.array(test_preds))) / len(test_preds))
        return accuracy, recall, loss
    elif type == 2:
        return ytest.ravel(), np.array(test_preds)
    else:
        return 0


if __name__ == "__main__":
    # retrieve training data
    id_set_train = (75397, 150448, 214861, 156709, 129629, 33078, 6744, 226034, 40190, 76253)
    results_train = "../data/train_search_results.csv"
    train_file = "../../data/train.jsonl"
    positives_train = positiveList(id_set_train, train_file, results_train)
    with open("../data/embeddings_train.jsonl", 'r') as openfile:
        embedding_train = json.loads(openfile.readlines()[0])

    # retrieve test data
    id_set_test = (137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708)
    results_test = "../data/test_search_results.csv"
    test_file = "../../data/shared_task_dev.jsonl"
    positives_test = positiveList(id_set_test, test_file, results_test)
    with open("../data/embeddings_test.jsonl", 'r') as openfile:
        embedding_test = json.loads(openfile.readlines()[0])

    # set training parameters
    learning_rate_list = np.append([0.01], np.arange(0.05, 0.95, 0.05))
    iterations_list = [2000, 4000, 5000, 6000, 8000]

    # train and test 100 times for each parameter combination
    writer = csv.writer(open("../data/T4_learn_params.csv", 'w'))
    writer.writerow(["learning rate", "epochs", "mean accuracy%",
                     "mean recall%", "mean loss"])
    for learning_rate in tqdm(learning_rate_list):
        for iterations in iterations_list:
            accuracy_list = np.array([])
            recall_list = np.array([])
            loss_list = np.array([])
            for _ in range(100):
                # retrieve randomised data
                x_train, y_train = data(embedding_train, positives_train)
                x_test, y_test = data(embedding_test, positives_test)

                # perform regression, get test results98
                accuracy, recall, loss = logisticRegression(
                    x_train, y_train, x_test, y_test, iterations, learning_rate)
                accuracy_list = np.append(accuracy_list, accuracy)
                recall_list = np.append(recall_list, recall)
                loss_list = np.append(loss_list, loss)
            # prepare output data
            avg_acc = np.average(accuracy_list) * 100
            std_acc = np.std(accuracy_list, ddof=1) * 100
            val_acc, uncert_acc = rounder(avg_acc, std_acc)
            str_acc = str(val_acc) + "±" + str(uncert_acc)

            avg_rec = np.average(recall_list) * 100
            std_rec = np.std(recall_list, ddof=1) * 100
            val_rec, uncert_rec = rounder(avg_rec, std_rec)
            str_rec = str(val_rec) + "±" + str(uncert_rec)

            avg_loss = np.round(np.average(loss_list), 2)

            # write to file
            writer.writerow([learning_rate, iterations,
                             str_acc, str_rec, avg_loss])
