import sys
sys.path.append('../../lib/')
from common_functions import *
from wikiparser import parseWiki
from tqdm import tqdm


def mySplit(s, delim=None):
    """
    split without empty elements at the ends
    :param s: string
    :param delim: delimeter for string splitting
    :return: array of strings
    """
    return [x for x in s.split(delim) if x]


def docSentences(data_arr, result_dir):
    """
    extracts all sentences corresponding to each document id
    :param data_arr: dictionary matching document name to it's lines
    :param result_dir: location of file containing search results
    :return: list of list of sentences
    """
    results = docIds(directory=result_dir)
    doc_text = [[[] for _ in range(5)] for _ in range(10)]
    for i in range(10):
        for j in range(5):
            text = data_arr[results[i][j]]
            doc_text[i][j].extend(text)
    return doc_text


def claimResultEmbeddings(embedding_dict, claims, wiki, result_dir):
    """
    creates a word embedding for results and claims
    :param embedding_dict: dictionary that matches a word to it's embedding vector
    :param claims: dictionary containing queries
    :param wiki: dictionary matching document name to it's lines
    :param result_dir: location of file containing search results
    :return: list of embeddings for each query and for each sentence in each result
    """
    results = docSentences(wiki, result_dir)
    embeddings = [[[] for _ in range(6)] for _ in range(10)]
    print("generating embeddings from " + str(result_dir) + " ...")
    for i in tqdm(range(10)):
        claim_vector = vectorise(embedding_dict, claims[i]['terms'])
        embeddings[i][0].extend(claim_vector)
        for j in range(5):
            if results[i][j]:
                for sentence in results[i][j]:
                    tokens = extractWordsB(sentence.lower())
                    sentence_vect = vectorise(embedding_dict, tokens)
                    embeddings[i][j+1].append(sentence_vect)
    return embeddings


if __name__ == "__main__":
    # get both sets of queries
    train_query_results = "../data/train_search_results.csv"
    queries_train = loadQueries(2, file="../../data/train.jsonl")
    test_query_results = "../data/test_search_results.csv"
    test_ids = (137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708)
    test_dir = "../../data/shared_task_dev.jsonl"
    queries_test = loadQueries(2, id_set=test_ids, file=test_dir)

    # import sentences
    id2line = parseWiki()

    # import word embeddings
    embedding_dictionary = loadEmbeddings('../../data/glove.6B.300d.txt')

    # store embeddings for claims and results into json file
    embeddings_train = claimResultEmbeddings(embedding_dictionary, queries_train, id2line, train_query_results)
    with open('../data/embeddings_train.jsonl', 'w') as outfile:
        json.dump(embeddings_train, outfile)
    embeddings_test = claimResultEmbeddings(embedding_dictionary, queries_test, id2line, test_query_results)
    with open('../data/embeddings_test.jsonl', 'w') as outfile:
        json.dump(embeddings_test, outfile)
