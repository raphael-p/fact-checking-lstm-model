import json
import re
import csv
import random
import math
import numpy as np
from tqdm import tqdm
# This file stores functions used in more than one file


def loadQueries(option, file,
                id_set=(75397, 150448, 214861, 156709,
                        129629, 33078, 6744, 226034,
                        40190, 76253)):
    """
    loads 10 queries from FEVER
    """
    data = []
    with open(file, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            if (json.loads(line)['id'] in id_set):
                data.append(json.loads(line))
    list = []
    for i in range(10):
        if (option == 1):
            terms = extractWordsA(data[i]['claim'].lower())
        elif (option == 2):
            terms = extractWordsB(data[i]['claim'].lower())
        else:
            raise ValueError('invalid extraction option. Choose 1 or 2')
        list.append({'id': data[i]['id'], 'terms': terms})
    return list


def extractWordsA(sample):
    """
    extracts words and numbers from a string
    :param sample: any given string
    :return: lowercased string, free of punctuation
    """
    list = []
    for word in sample.split():
        try:
            word = re.sub('[^a-z^0-9]+', "", word.replace(
                "'s", "").replace("'t ", "t "))
            if (word):
                list.append(word)
        except UnicodeEncodeError:
            continue
    return list


def extractWordsB(sample):
    """
    extracts words and numbers from a string
    :param sample: any given string
    :return: lowercased string, free of punctuation
    """
    list = []
    contractions = [["n't", " n't"], ["'s", " 's"], ["'m", " 'm"], ["'re", " 're"], ["'ve", " 've"], ["'ll", " 'll"],
                    ["'d", " 'd"]]
    for word in sample.split():
        try:
            for contraction in contractions:
                word = word.replace(contraction[0], contraction[1])
            word = re.findall(r"[\w']+|[.,!?;-]", word)
            if word:
                list.extend(word)
        except UnicodeEncodeError:
            continue
    return list


def uniqueList(queries):
    """
    returns a list of all unique terms from all queries
    :param queries: array containing all query dictionaries
    """
    uniques = []
    for query in queries:
        for term in query['terms']:
            if term not in uniques:
                uniques.append(term)
    return uniques


def uniqueListSingle(query, term_to_index):
    """
    transform query into an array of unique term keys
    :param query: dictionary containing an array of a query's terms
    :param term_to_index: matches a query term to it's index
    :return: the list of the unique terms used in the query
    """
    uniques = []
    for term in query['terms']:
        if term not in uniques:
            uniques.append(term_to_index[term])
    return uniques


def trim(string):
    """
    makes strings more readable by replacing a few common tags with the punctuation they represent
    :param string: a string from the database
    :return: simplified string
    """
    trimmed = string.replace("-LRB-", "(").replace("-RRB-", ")").replace("-COLON-", ":")
    return trimmed


def untrim(string):
    """
    revert trimmed strings (see trim function)
    :param string: a string from the database
    :return: untrimmed string
    """
    untrimmed = string.replace("(", "-LRB-").replace(")", "-RRB-").replace(":", "-COLON-")
    return untrimmed


def docIds(directory="../task_4/data/train_search_results.csv"):
    """
    returns ids of 5 the results for each of the 10 claims
    :param directory: location of file containing search results
    :return: list of document id lists
    """
    doc_ids = []
    with open(directory, 'rt') as f:
        reader = csv.reader(f)
        for result in reader:
            if result[0] == 'claim id':
                continue
            doc_ids.append([result[1], result[2], result[3], result[4],
                            result[5]])
    return doc_ids


def rounder(value, uncertainty):
    """
    rounds a value and its uncertainty
    according to scientific convention
    :param value: a given measure
    :param uncertainty: error on value
    :return: rounded value and error
    """
    if uncertainty < 1:
        decimal = abs(int(math.floor(math.log10(uncertainty))))
        value = np.round(value, decimal)
        uncertainty = np.round(uncertainty, decimal)
    elif uncertainty == 0:
        value = int(np.round(value))
    else:
        value = int(np.round(value))
        uncertainty = int(np.round(uncertainty))
    return value, uncertainty


def decision(probability):
    """
    probabilistic coin flip
    :param probability: likelihood of returning True
    :return: True or False
    """
    return random.random() < probability


def vectorise(embedding_dict, terms):
    """
    generates word embeddings, adds them together and normalises
    :param embedding_dict: glove data, matches term to embedding
    :param terms: list of terms to be embedded
    :return: list containing an embedding
    """
    embed_vector = np.zeros(len(embedding_dict['the']))
    for i in range(0, len(terms)):
        try:
            embed_vector += np.array(embedding_dict[terms[i]])
        except KeyError:
            # print("not found: "+str(terms[i]))
            continue
    if not all(embed_vector) == 0:
        embed_vector = embed_vector / np.linalg.norm(embed_vector)
    return list(embed_vector)

def vectoriseB(embedding_dict, terms):
    """
    creates a vector of word embeddings for a given set of terms
    :param embedding_dict: glove data, matches term to embedding
    :param terms: list of terms to be embedded
    :return: list containing an embedding
    """
    embeddings = []
    for i in range(0, len(terms)):
        try:
            embeddings.append(embedding_dict[terms[i]])
        except KeyError:
            # print("not found: "+str(terms[i]))
            continue
    return list(embeddings)


def loadEmbeddings(filename):
    """
    loads GLOVE word embeddings from file
    :param filename: name of file containing word embeddings
    :return: dictionary of word embeddings
    """
    embedding_dict = {}
    file = open(filename, 'r', encoding='UTF-8')
    print('loading GLOVE...')
    for line in tqdm(file.readlines()):
        row = line.strip().split(' ')
        vocab_word = row[0]
        embed_vector = [float(i) for i in row[1:]]  # convert to list of float
        embedding_dict[vocab_word] = embed_vector

    file.close()
    return embedding_dict
