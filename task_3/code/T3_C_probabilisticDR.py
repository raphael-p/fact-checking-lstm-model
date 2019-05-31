import glob
import os
import sys
sys.path.append('../../lib/')
from common_functions import *


def wordCountEstimate(maxDoc):
    """
    method used to estimate how many terms are in the collection
    :param maxDoc: size of document sample
    :return: number of words in sample
    """
    uniques = []
    unique_count = 0
    total_count = 0
    for filename in glob.glob(os.path.join("../../data/wiki-pages", '*.jsonl')):
        with open(filename, 'r') as openfile:
            for iline, line in enumerate(openfile.readlines()):
                doc_text = extractWordsA(json.loads(line)['text'].lower())
                for word in doc_text:
                    total_count += 1
                if iline+1 >= maxDoc:
                    return total_count
    return 0


def queryProb(indexed_doc, terms):
    """
    Calculates document search score using basic PDR
    :param indexed_doc: inverse index of a document
    :param terms: id numbers of terms in query
    :return: basic score of the document relative to the query
    """
    doc = indexed_doc.split()
    D = int(doc[1])
    if len(doc) < (len(terms) * 2 + 2):
        return 0
    index_arr = {int(doc[2 * i]): int(doc[2 * i + 1])/D
                 for i in range(1, int(len(doc)/2))}
    product = 1
    for term in terms:
        if term in index_arr:
            product *= index_arr[term]
        else:
            return 0
    return product


def laplaceProb(indexed_doc, terms, V):
    """
    Calculates document search score using laplace smoothing
    :param indexed_doc: inverse index of a document
    :param terms: id numbers of terms in query
    :param V: vocabulary size
    :return: laplace score of the document relative to the query
    """
    doc = indexed_doc.split()
    eps = 1e-4
    if len(doc) < 5:
        return 0
    D = int(doc[1])
    index_arr = {int(doc[2 * i]): int(doc[2 * i + 1])
                 for i in range(1, int(len(doc)/2))}
    product = 1
    for term in terms:
        if term in index_arr:
            product *= (index_arr[term] + 1) / (D + (V*eps))
        else:
            product *= eps / (D + (V*eps))

    return product


def interpolProb(indexed_doc, terms, p_coll, type, lmu):
    """
    Calculates document search score using interpolation probability
    methods
    :param indexed_doc: inverse index of a document
    :param terms: id numbers of terms in query
    :param p_coll: array of probability of each query
    term to occur on the collection
    :param type: smoothing type, 'dirichlet' or 'jm'
    (jelinek-mercer)
    :param lmu: lamdba or mu value, depending on smoothing type
    :return: interpolation score of the document relative to the query
    """
    doc = indexed_doc.split()
    if len(doc) < 5:
        return 0
    D = int(doc[1])
    if type == 'dirichlet':
        lam = D / (D + lmu)
    elif type == 'jm':
        lam = lmu
    index_arr = {int(doc[2 * i]): int(doc[2 * i + 1])
                 for i in range(1, int(len(doc)/2))}
    product = 1
    for term in terms:
        p_in_D = 0
        p_in_C = p_coll[term]
        if term in index_arr:
            p_in_D = index_arr[term] / D
        product *= (lam * p_in_D + (1 - lam) * p_in_C)
    return product


def engine(q_ids, index, V, p_coll, lam, mu):
    """
    performs searches for 4 different methods
    :param q_ids: id numbers of unique term in the query
    :param index: inverted index of collection
    :param V: vocabulary size
    :param p_coll: array of probability of each query
    term to occur on the collection
    :param lam: value of lambda for jm smoothing
    :param mu: value of mu for dirichlet smoothing
    :return: 5 document ids (search results) per smoothing type (4)
    """
    results = [[[None, 0]] * 5 for _ in range(4)]
    min_score = [0] * 4
    for document in index:
        doc_id = document.split()[0]
        score_bas = queryProb(document, q_ids)
        score_lap = laplaceProb(document, q_ids, V)
        score_jm = interpolProb(document, q_ids, p_coll, 'jm', lam)
        score_dir = interpolProb(document, q_ids, p_coll, 'dirichlet', mu)
        scores = [score_bas, score_lap, score_jm, score_dir]
        for i in range(len(scores)):
            score = scores[i]
            result = results[i]
            # weed out weak queries
            if not score:
                continue
            if score < min_score[i]:
                continue

            # input relevant query into list
            for rank in range(5):
                if score > result[rank][1]:
                    result.insert(rank, [doc_id, score])
                    min_score[i] = result[4][1]
                    results[i] = result[:5]
                    break
    print(results)
    return results


if __name__ == "__main__":
    # estimating vocabulary and collection size using wordCountEstimate()
    docNum = 5416536  # number of documents
    sample_size = 27000
    sample_words = 2301569
    # to estimate vocabulary size, assume that as N->inf, vocab->500 * sqrt(N)
    V = int(500 * np.sqrt(docNum))
    C = int(sample_words * (docNum/sample_size))  # estimate of collection size
    avg_doc_length = int(C / docNum)

    # generating queries
    queries = loadQueries(1, file="../../data/train.jsonl")
    unique_terms = uniqueList(queries)
    term_to_index = {unique_terms[i]: i for i in range(len(unique_terms))}
    index_to_term = {i: unique_terms[i] for i in range(len(unique_terms))}

    # loading inverse index
    with open("../../task_2/data/collection_index.jsonl", 'r') as openfile:
        inv_index = json.loads(openfile.readlines()[0])

    # loading index of probabilities of each search term in collection
    with open("../data/collection_prob.jsonl", 'r') as openfile:
        p_coll = json.loads(openfile.readlines()[0])

    # running all methods, and storing to csv
    writer_u = csv.writer(open("../data/unigram.csv", 'w'))
    writer_l = csv.writer(open("../data/laplace.csv", 'w'))
    writer_jm = csv.writer(open("../data/jelinek_mercer.csv", 'w'))
    writer_d = csv.writer(open("../data/dirichlet.csv", 'w'))
    writer_list = [writer_u, writer_l, writer_jm, writer_d]
    for writer in writer_list:
        writer.writerow(["claim id", "doc id_1", "doc id_2", "doc id_3",
                         "doc id_4", "doc id_5"])
    for query in queries:
        print(" ".join(query['terms']))
        query_ids = [term_to_index[term] for term in query['terms']]
        combined_search_results = engine(query_ids, inv_index, V=V,
                                         p_coll=p_coll, lam=0.4,
                                         mu=avg_doc_length)
        for i in range(len(writer_list)):
            result_line = [query['id']]
            search_results = combined_search_results[i]
            for result in search_results:
                if result[0]:
                    result[0] = untrim(result[0])
                result_line.append(result[0])
            writer_list[i].writerow(result_line)
