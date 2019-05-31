import sys
sys.path.append('../../lib/')
from common_functions import *


def interpolProb(indexed_doc, terms, p_coll, lam):
    """
    Calculates document search score using jelinek-mercer
    smoothing
    :param indexed_doc: inverse index of a document
    :param terms: id numbers of terms in query
    :param p_coll: array of probability of each query
    term to occur on the collection
    :param lam: lamdba value for jm
    :return: interpolation score of the document relative to the query
    """
    doc = indexed_doc.split()
    if len(doc) < 5:
        return 0
    D = int(doc[1])
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


def engine(q_ids, index, p_coll, lam):
    """
    performs searches for jelinek-mercer smoothing
    :param q_ids: id numbers of unique term in the query
    :param index: inverted index of collection
    :param p_coll: array of probability of each query
    term to occur on the collection
    :param lam: value of lambda for jm smoothing
    :return: 5 document ids (search results)
    """
    results = [[None, 0] for _ in range(5)]
    min_score = 0
    for document in index:
        doc_id = document.split()[0]
        score = interpolProb(document, q_ids, p_coll, lam)
        # weed out weak queries
        if not score:
            continue
        if score < min_score:
            continue
        # input relevant query into list
        for rank in range(5):
            if score > results[rank][1]:
                results.insert(rank, [doc_id, score])
                min_score = results[4][1]
                results = results[:5]
                break
    print(results)
    return results


if __name__ == "__main__":
    # generating queries
    queries = loadQueries(1, file="../../data/train.jsonl")
    unique_terms = uniqueList(queries)
    term_to_index = {unique_terms[i]: i for i in range(len(unique_terms))}

    # loading inverse index
    with open("../../task_2/data/collection_index.jsonl", 'r') as openfile:
        inv_index = json.loads(openfile.readlines()[0])

    # loading index of probabilities of each search term in collection
    with open("../data/collection_prob.jsonl", 'r') as openfile:
        p_coll = json.loads(openfile.readlines()[0])

    # Jelinek-Mercer smoothing: testing for several lambdas
    writer = csv.writer(open("../data/labdTest.csv", 'w'))
    writer.writerow(["claim id", "lambda", "doc id_1", "doc id_2",
                    "doc id_3", "doc id_4", "doc id_5"])
    labd_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for labd in labd_arr:
        for query in queries:
            print(" ".join(query['terms']))
            query_ids = [term_to_index[term] for term in query['terms']]
            search_results = engine(query_ids, inv_index, p_coll, labd)
            result_line = [query['id'], labd]
            for result in search_results:
                result_line.append(result[0])
            writer.writerow(result_line)
