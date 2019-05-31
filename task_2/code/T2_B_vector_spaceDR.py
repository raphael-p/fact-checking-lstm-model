import sys
sys.path.append('../../lib/')
from common_functions import *


def inverseDocFreq(inverse_index, num_unique):
    """
    creates array of how many documents query terms occur in
    """
    N = len(inverse_index)
    doc_count = np.zeros(num_unique)
    print("calculating inverse document frequency ...")
    for document in tqdm(inverse_index):
        doc_arr = document.split()
        for i in range(2, len(doc_arr)):
            if i % 2 == 0:
                doc_count[int(doc_arr[i])] += 1
    freqs = np.log(N/doc_count)
    return freqs


def queryRater(query, idfs, inverse_index):
    """
    calculates cosine-normalised rating of each document for each query term,
    returns top 5 documents
    :param query: array of indices of the terms found in the query
    :param idfs: array of inverse document frequency for each unique term
    (accross all queries)
    :param inverse_index: the inverted index of the wiki-pages database
    :return: an array containing the id's five best documents for a query,
     along with their score
    """
    results = [[None, 0]]*5
    min_score = 0
    for document in inverse_index:
        doc_arr = document.split()
        doc_id = doc_arr[0]
        doc_len = int(doc_arr[1])
        if not doc_len > 1:
            continue
        weights = np.array([])
        sum_idf_tots = 0
        sum_idf_founds = 0
        sum_idf_tot = 0
        sum_idf_found = 0
        for term in query:
            idf = idfs[term]
            sum_idf_tot += idf**2
            sum_idf_tots += idf
            tf = 0
            for i in range(2, len(doc_arr)):
                if i % 2 == 0:
                    if term == int(doc_arr[i]):
                        tf = int(doc_arr[i+1])
            if not tf:
                continue
            value = (tf/np.log(doc_len)) * idf
            weights = np.append(weights, value)
            sum_idf_found += idf**2
            sum_idf_founds += idf
        idf_percentage = sum_idf_found / sum_idf_tot
        if not sum_idf_founds/sum_idf_tots > 0.5:
            continue
        score = np.sum(weights) * idf_percentage
        if score < min_score:
            continue
        for rank in range(5):
            if score > results[rank][1]:
                results.insert(rank, [doc_id, score])
                min_score = results[4][1]
                break
    return results[:5]


if __name__ == '__main__':
    # retrieve queries, extract relevant information
    queries = loadQueries(1, file="../../data/train.jsonl")
    unique_terms = uniqueList(queries)
    num_unique = len(unique_terms)
    term_to_index = {unique_terms[i]: i for i in range(len(unique_terms))}

    # retrieve inverse index, calculate inverse document frequencies
    with open("../data/collection_index.jsonl", 'r') as openfile:
        inv_index = json.loads(openfile.readlines()[0])
    inv_doc_freq = inverseDocFreq(inv_index, num_unique)

    # generate search results
    writer = csv.writer(open("../data/vector_spaceDR.csv", 'w'))
    writer.writerow(["claim id", "doc id_1", "doc id_2", "doc id_3",
                    "doc id_4", "doc id_5"])
    for query in queries:
        query_indices = uniqueListSingle(query, term_to_index)
        searchResults = queryRater(query_indices, inv_doc_freq, inv_index)
        result_line = [query['id']]
        for result in searchResults:
            if result[0]:
                result[0] = untrim(result[0])
            result_line.append(result[0])
        writer.writerow(result_line)
