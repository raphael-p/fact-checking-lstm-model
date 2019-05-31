import sys
sys.path.append('../../lib/')
from common_functions import *


def pWordInC(inverse_index, num_unique, collection_size):
    """
    creates array of how likely it is to encounter each query term
    in the collection
    :param inverse_index: inverse index of a document
    :param num_unique: number of unique query terms
    :param collection_size: number of words in collection
    :return: array of probabilities
    """
    term_count = np.zeros(num_unique)
    for document in inverse_index:
        doc_arr = document.split()
        for i in range(2, len(doc_arr)):
            if i % 2 == 0:
                term_count[int(doc_arr[i])] += int(doc_arr[i+1])
    p_W_C = term_count/collection_size
    return p_W_C


if __name__ == "__main__":
    # estimating vocabulary and collection size using wordCountEstimate():
    docNum = 5416536  # number of documents
    sample_size = 27000
    sample_words = 2301569
    C = int(sample_words * (docNum/sample_size))  # estimate of collection size

    # generating queries:
    queries = loadQueries(1, file="../../data/train.jsonl")
    unique_terms = uniqueList(queries)

    # loading inverse index
    with open("../../task_2/data/collection_index.jsonl", 'r') as openfile:
        inv_index = json.loads(openfile.readlines()[0])

    # creating index of probabilities of each search term in collection:
    p_coll = pWordInC(inv_index, len(unique_terms), C)
    with open('../data/collection_prob.jsonl', 'w') as outfile:
        json.dump(p_coll.tolist(), outfile)