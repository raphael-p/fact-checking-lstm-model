import sys
sys.path.append('../../lib/')
from common_functions import *
sys.path.append('../../task_2/code/')
from T2_A_inverse_index import inverseIndexing
from T2_B_vector_spaceDR import queryRater, inverseDocFreq

if __name__ == "__main__":
    # retrieve queries, extract relevant information
    test_ids = (137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708)
    test_dir = "../../data/shared_task_dev.jsonl"
    queries = loadQueries(1, id_set=test_ids, file=test_dir)
    unique_terms = uniqueList(queries)
    num_unique = len(unique_terms)
    term_to_index = {unique_terms[i]: i for i in range(len(unique_terms))}

    # create inverse index based on new queries
    collection_index = inverseIndexing(unique_terms, term_to_index)

    # generate inverse document frequency vector
    inv_doc_freq = inverseDocFreq(collection_index, num_unique)

    # get and store results
    writer = csv.writer(open("../data/test_search_results_auto.csv", 'w'))
    writer.writerow(["claim id", "doc id_1", "doc id_2", "doc id_3",
                    "doc id_4", "doc id_5"])
    for query in tqdm(queries):
        query_indices = uniqueListSingle(query, term_to_index)
        searchResults = queryRater(query_indices, inv_doc_freq, collection_index)
        result_line = [query['id']]
        for result in searchResults:
            if result[0]:
                result[0] = untrim(result[0])
            result_line.append(result[0])
        writer.writerow(result_line)
