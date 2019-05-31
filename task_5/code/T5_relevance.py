import sys
sys.path.append('../../task_4/code/')
from T4_C_log_res import *
sys.path.append('../../lib/')
from common_functions import *


def recall(actual, output):
    """
    recall metric
    :param actual: label data
    :param output: neural network output
    :return: recall score for this network
    """
    recall_count = 0
    for j in range(actual.size):
        if (actual[j] == 1) and (output[j] == 1):
            recall_count = recall_count + 1
    return recall_count / np.sum(actual)


def precision(actual, output):
    """
    precision metric
    :param actual: label data
    :param output: neural network output
    :return: precision score for this network
    """
    true_pos = 0
    for i in range(len(output)):
        if actual[i] == output[i] and actual[i] == 1:
            true_pos = true_pos + 1
    if np.sum(output) == 0:
        return 0
    else:
        return true_pos / np.sum(output)


def fmeasure(rec, prec, coef):
    """
    F1 metric
    :param rec: recall score for this network
    :param prec: precision score for this network
    :param coef: relative weight given to precision
    :return: F1 score for this network
    """
    denom = coef * (1/prec) + (1-coef) * (1/rec)
    return 1/denom


if __name__ == "__main__":
    # retrieve training data
    id_set_train = (75397, 150448, 214861, 156709, 129629, 33078, 6744, 226034, 40190, 76253)
    results_train = "../../task_4/data/train_search_results.csv"
    train_file = "../../data/train.jsonl"
    positives_train = positiveList(id_set_train, train_file, results_train)
    with open("../../task_4/data/embeddings_train.jsonl", 'r') as openfile:
        embedding_train = json.loads(openfile.readlines()[0])

    # retrieve test data
    id_set_test = (137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708)
    results_test = "../../task_4/data/test_search_results.csv"
    test_file = "../../data/shared_task_dev.jsonl"
    positives_test = positiveList(id_set_test, test_file, results_test)
    with open("../../task_4/data/embeddings_test.jsonl", 'r') as openfile:
        embedding_test = json.loads(openfile.readlines()[0])

    learning_rate_list = np.append([0.01], np.arange(0.1, 0.95, 0.1))
    iterations_list = [8000, 8000, 6000, 4000, 6000, 2000, 8000, 4000, 2000, 4000]

    recall_list = []
    precision_list = []
    writer = csv.writer(open("../data/metrics.csv", 'w'))
    writer.writerow(["learning rate", "epochs", "recall%",
                     "precision%", "F1%"])
    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        iterations = iterations_list[i]
        for _ in range(50):
            # retrieve randomised data
            x_train, y_train = data(embedding_train, positives_train)
            x_test, y_test = data(embedding_test, positives_test)

            # perform regression, get test results
            y, y_tilde = logisticRegression(
                x_train, y_train, x_test, y_test, iterations, learning_rate, type=2)
            recall_list.append(recall(y, y_tilde))
            precision_list.append(precision(y, y_tilde))

        # calculating metrics
        alpha = 0.5
        avg_prec = np.average(precision_list)
        std_prec = np.std(precision_list)
        val_prec, uncert_prec = rounder(avg_prec * 100, std_prec * 100)
        str_prec = str(val_prec) + "±" + str(uncert_prec)

        avg_rec = np.average(recall_list)
        std_rec = np.std(recall_list)
        val_rec, uncert_rec = rounder(avg_rec * 100, std_rec * 100)
        str_rec = str(val_rec) + "±" + str(uncert_rec)

        avg_f1 = np.round(fmeasure(avg_rec, avg_prec, alpha), 2) * 100

        # printing and storing metrics
        writer.writerow([str(np.round(learning_rate,2)),
                         str(iterations),
                         str_rec, str_prec, str(avg_f1)])
        print("learning rate: " + str(np.round(learning_rate,2))
              + " epochs: " + str(iterations)
              + "\n\t recall: " + str_rec + "%"
              + " precision: " + str_prec + "%"
              + " F1 score : " + str(avg_f1) + "%")
