import json
import numpy as np
import glob
import os
import re
from collections import Counter
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import zetac
from tqdm import tqdm


def loadDataset(path, instance_num=1e6):
    """
    Reads the Fever Training set, returns list of examples.
    instance_num: how many examples to load. Useful for debugging.
    """
    data = []
    for filename in tqdm(glob.glob(os.path.join(path, '*.jsonl'))):
        with open(filename, 'r') as openfile:
            for iline, line in enumerate(openfile.readlines()):
                data.append(json.loads(line)['text'].lower())
                if iline+1 >= instance_num:
                    break
    return data


def extractWords(data):
    """
    transforms document into array of words ready to be analysed
    """
    list = []
    for sample in data:
        for word in sample.split():
            try:
                word = re.sub('[^a-z\ \']+', " ", word)
                word = re.sub('[^a-z]+', "", word)
                if (word):
                    list.append(word)
            except UnicodeEncodeError:
                continue
    return list


def zipfPlot(zipfs_data, N, filename):
    """
    uses list of words and plots a zip's law graph for the N most
    frequents ones
    :param zipfs_data: list of 5000 most frequent terms in collection and
    their frequencies
    :param N: number of terms to include in model and fit
    :param filename: name of file to save the plot to
    """
    freq_arr = zipfs_data[:N]
    x_axis = np.arange(1, N+1, 1)
    result = curve_fit(zipfFun, x_axis, freq_arr)
    a = result[0][0]
    b = result[0][1]
    fit_curve = x_axis**(-a) / zetac(b)
    plt.bar(x_axis, freq_arr, label="document data")
    plt.plot(x_axis, fit_curve, 'r-', label="fitted line", linewidth=0.5)
    string = ("fitting along zipf's law:"
              + "\nexponent parameter, s = {0:f}"
              + "\nnormalisation parameter, n = {1:f}"
              + "\n $f = 1/k^s * 1/\zeta(n)$, where k is the rank"
              ).format(a, b)
    plt.text(25, 2*10**7, string)
    plt.title("Zipf's Law, {} most frequent words in corpus".format(N))
    plt.xlabel("word rank")
    plt.ylabel("word frequency")
    plt.legend()
    plt.savefig(filename+'.png', dpi=400)
    return


def zipfFun(x, a, b):
    return (x**-a)/zetac(b)


if __name__ == "__main__":
    N = 400

    # Generate array of the N most frequent terms in the collection
    train_path = "../../data/wiki-pages"
    train_data = loadDataset(path=train_path)
    train_list = extractWords(train_data)
    freq_list = Counter(train_list)
    top = freq_list.most_common(N)

    # Extract most frequent terms, create a plot and fit to a function
    data = np.zeros(N)
    for i in range(N):
        data[i] = top[i][1]
    zipfPlot(data, N, "data/text_statistics")
