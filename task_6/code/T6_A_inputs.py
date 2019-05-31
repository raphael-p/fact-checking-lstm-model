import unicodedata
import sys
import numpy as np
sys.path.append('../../lib')
from common_functions import *
from wikiparser import parseWiki


def dataPrep(filename, docs, emb_dict, max_evi, type="sequence"):
    """
    extracts necessary data for machine learning from
    fever jsonl labelled datasets
    :param filename: file from which to source data
    :param docs: dictionary matching document name to it's lines
    :param emb_dict: dictionary that matches a word to it's embedding vector
    :param max_evi: maximum number of allowed evidences per claim
    :param type: how to store data
    sequence -> claim embedding followed by evidence embeddings
    pair -> claim/evidence embedding pairs
    :return: inputs and labels for machine learning
    """
    instance_num = 1000000
    x = []
    y = []
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
            claim_emb = vectorise(emb_dict, claim)
            evi_pairs = []
            if type == "sequence":
                embeddings = [claim_emb]
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
                        sentence_emb = vectorise(emb_dict, sentence)
                        if type == "sequence":
                            embeddings.append(sentence_emb)
                        elif type == "pair":
                            x.append(claim_emb + sentence_emb)
                            y.append(input_type)
                    except KeyError:
                        continue
            if type == "sequence":
                embeddings = padding(embeddings, max_evi)
                x.append(embeddings)
                y.append(input_type)
            if iline + 1 >= instance_num:
                break
    return x, y


def padding(embeddings, max_num):
    """
    sets array of embeddings to desired size by trimming or padding
    :param embeddings: array of embeddings
    :param max_num: desired number of embeddings
    :return: embedding array of size max_num
    """
    emb_length = len(embeddings[0])
    n_emb = len(embeddings) - 1
    n_pad = max_num - n_emb
    if n_pad == 0:
        return embeddings
    elif n_pad > 0:
        for _ in range(n_pad):
            embeddings.append([0] * emb_length)
        return embeddings
    else:
        return embeddings[:(len(embeddings)+n_pad)]


if __name__ == "__main__":
    # define file paths
    train_file = "../../data/train.jsonl"
    dev_file = "../../data/shared_task_dev.jsonl"

    # import wiki sentences and embeddings
    wiki = parseWiki()
    embedding_dictionary = loadEmbeddings('../../data/glove.6B.300d.txt')

    # generate data
    timesteps = 4  # (timesteps-1) for LSTM
    x_train, y_train = dataPrep(train_file, wiki, embedding_dictionary, timesteps)
    x_test, y_test = dataPrep(dev_file, wiki, embedding_dictionary, timesteps)

    # TEMPORARILY STORING RESULTS
    with open('../data/x_train.jsonl', 'w') as outfile:
        json.dump(x_train, outfile)
    with open('../data/y_train.jsonl', 'w') as outfile:
        json.dump(y_train, outfile)
    with open('../data/x_test.jsonl', 'w') as outfile:
        json.dump(x_test, outfile)
    with open('../data/y_test.jsonl', 'w') as outfile:
        json.dump(y_test, outfile)
    print(np.array(y_test).shape)
    print(np.array(x_test).shape)
    print("done.")
