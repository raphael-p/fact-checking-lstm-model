import unicodedata
import sys
import numpy as np
sys.path.append('../../lib')
from common_functions import *
from wikiparser import parseWiki


def dataPrep(filename, docs, instance_num=1000000):
    """
    extracts necessary data for machine learning from
    fever jsonl labelled datasets
    :param filename: file from which to source data
    :param docs: dictionary matching document name to it's lines
    :param instance_num: maximum number of lines to read from file
    :return: tokenised claim and evidence for training and testing, and associated labels
    """
    claims = []
    sentences = []
    labels = []

    with open(filename, 'r') as openfile:
        for iline, line in tqdm(enumerate(openfile.readlines())):
            label = json.loads(line)['label']
            if label == "SUPPORTS":
                input_type = 1
            elif label == "REFUTES":
                input_type = 0
            else:
                continue
            claim = extractWordsB(json.loads(line)['claim'].lower())
            evi_pairs = []
            claim_evi = []
            for evidences in json.loads(line)["evidence"]:
                for evidence in evidences:
                    try:
                        doc_id = unicodedata.normalize('NFC', evidence[2])
                        sent_pos = evidence[3]
                        pair = [doc_id, sent_pos]
                        if pair in evi_pairs:
                            continue
                        evi_pairs.append(pair)
                        doc = docs[doc_id]
                        sentence = extractWordsB(doc[sent_pos].lower())
                        claim_evi.extend(sentence)
                    except KeyError:
                        continue
            if len(evidence) < 1:
                continue
            claims.append(claim)
            sentences.append(claim_evi)
            labels.append(input_type)
            if iline + 1 >= instance_num:
                break
    return claims, sentences, labels


if __name__ == "__main__":
    # define file paths
    train_file = "../../data/train.jsonl"
    dev_file = "../../data/shared_task_dev.jsonl"

    # import wiki sentences and embeddings
    wiki = parseWiki()

    # generate data
    claim_train, evi_train, y_train = dataPrep(train_file, wiki)
    claim_test, evi_test, y_test = dataPrep(dev_file, wiki)
    claims = claim_train + claim_test
    evis = evi_train + evi_test
    labels = y_train + y_test

    # TEMPORARILY STORING RESULTS
    with open('../data/claims.jsonl', 'w') as outfile:
        json.dump(claims, outfile)
    with open('../data/evis.jsonl', 'w') as outfile:
        json.dump(evis, outfile)
    with open('../data/labels.jsonl', 'w') as outfile:
        json.dump(labels, outfile)
    print(np.array(claims).shape)
    print(np.array(evis).shape)
    print(np.array(labels).shape)
    print("done.")
    exit(0)