import os
import glob
import sys
sys.path.append('../../lib/')
from common_functions import *


def inverseIndexing(unique_terms, term_keys):
    """
    Creates an inverse index from the database
    :param unique_terms: array of unique query terms from the 10 queries
    :param term_keys: maps a term to its index
    :return: the inverse index to be stored in a jsonl file
    """
    index = []
    length = len(unique_terms)
    print("creating inverse index ...")
    for filename in tqdm(glob.glob(os.path.join("../../data/wiki-pages", '*.jsonl'))):
        with open(filename, 'r') as openfile:
            for iline, line in enumerate(openfile.readlines()):
                line = json.loads(line)
                doc_id = trim(line["id"])
                if doc_id:
                    doc_index = [0] * length
                    text = extractWordsA(line['text'].lower())
                    doc_length = len(text)
                    for word in text:
                        if word in unique_terms:
                            doc_index[term_keys[word]] += 1
                    doc_string = doc_id+" "+str(doc_length)
                    for i in range(length):
                        if doc_index[i]:
                            doc_string = doc_string+" "+str(i)+" "+str(
                                doc_index[i])
                    index.append(doc_string)
    return index


if __name__ == '__main__':
    queries = loadQueries(1, file="../../data/train.jsonl")
    unique_terms = uniqueList(queries)
    term_to_index = {unique_terms[i]: i for i in range(len(unique_terms))}

    # Generate inverse index of collection
    collection_index = inverseIndexing(unique_terms, term_to_index)
    with open('../data/collection_index.jsonl', 'w') as outfile:
        json.dump(collection_index, outfile)
