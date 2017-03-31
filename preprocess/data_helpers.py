import numpy as np
import re
import csv
from collections import namedtuple
from typing import Dict
import os
from preprocess.file_utils import deserialize, serialize
from preprocess.lcs_match import get_lcs_ratio
from app.decorator import exe_time

PAD_ID = 0
NULL_ID = 1
data_dir = os.path.join('..', 'dataset')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`-]", " ", string)
    string = re.sub(r"'s", " 's", string)
    string = re.sub(r"'ve", " have", string)
    string = re.sub(r"n't", " not", string)
    string = re.sub(r"'re", " are", string)
    string = re.sub(r"'m", " am", string)
    string = re.sub(r"'d", " would", string)
    string = re.sub(r"'ll", " will", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split(' ')


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data(path, need_label=True):
    x = []
    y = []
    with open(path) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        Row = namedtuple('Row', headers)
        for col_names in f_csv:
            row = Row(*col_names)
            x.append([row.question1, row.question2])
            if need_label:
                y.append(int(row.is_duplicate))
    return (x, y) if need_label else x


def vectorize_sent(sent, word2idx: Dict, oov_size, idx, sent_size=50):
    if idx == 0:
        trim_sent = sent[:sent_size - 1]
    else:
        trim_sent = sent[:sent_size]
    dict_len = len(word2idx)
    sent_ids = [word2idx.get(w, hash(w) % oov_size + dict_len) for w in trim_sent]

    # Add null token in sent1
    if idx == 0:
        sent_ids.append(NULL_ID)

    pad_length = sent_size - len(sent_ids)
    sent_ids.extend([PAD_ID] * pad_length)
    return sent_ids


def cast_y(ys):
    return [float(y) for y in ys]


def sample_eval_data(dev_x, dev_y, size):
    x_arr = np.array(dev_x)
    y_arr = np.array(dev_y)
    shuffle_indices = np.random.permutation(len(dev_x))
    return x_arr[shuffle_indices[:size]], y_arr[shuffle_indices[:size]]


def transform_y(x, y, alpha):
    for index in range(len(y)):
        lcs_ratio = get_lcs_ratio(*x[index])
        y[index] = alpha * y[index] + (1 - alpha) * lcs_ratio
    return y


def remove_lcs(x):
    x_new = []
    for sent1, sent2 in x:
        word_set1 = set(sent1)
        word_set2 = set(sent2)
        common_word_set = word_set1 & word_set2
        sent1_new = list(filter(lambda w: w not in common_word_set, sent1))
        sent2_new = list(filter(lambda w: w not in common_word_set, sent2))
        if len(sent1_new) == 0:
            sent1_new.append('?')
        if len(sent2_new) == 0:
            sent2_new.append('?')
        x_new.append([sent1_new, sent2_new])
    return x_new


def unpack_x_batch(x_batch):
    sents1, sents2 = [], []
    for s in x_batch:
        sents1.append(s[0])
        sents2.append(s[1])
    return sents1, sents2


def get_idf_ratio(a, b, idf_dict):
    common_ids = list(set(a) & set(b))
    common_weights = sum([idf_dict.get(x, 0) for x in common_ids])
    all_sents = a + b
    all_weights = sum([idf_dict.get(x, 0) for x in all_sents])
    return common_weights / (all_weights + 1e-3)



def get_extra_features(sents1, sents2, idf_dict):
    s1 = [list(filter(lambda x: x != PAD_ID and x != NULL_ID, s)) for s in sents1]
    s2 = [list(filter(lambda x: x != PAD_ID, s)) for s in sents2]
    lcs_ratios = [get_lcs_ratio(a, b) for a, b in zip(s1, s2)]
    idf_ratios = [get_idf_ratio(a, b, idf_dict) for a, b in zip(s1, s2)]

    extra_features = [[a, b] for a, b in
                      zip(lcs_ratios, idf_ratios)]
    return extra_features


def main():
    train_x, train_y = deserialize(os.path.join(data_dir, 'train.bin'))
    dev_x, dev_y = deserialize(os.path.join(data_dir, 'dev.bin'))

    word2index = deserialize(os.path.join(data_dir, 'word2index_glove.bin'))
    word_embeddings = deserialize(os.path.join(data_dir, 'word_embeddings_glove.bin'))
    oov_embed_size = len(word_embeddings) - len(word2index)

    # # train_x = remove_lcs(train_x)
    # train_ids = [[vectorize_sent(q, word2index, oov_embed_size, i, 50) for i, q in enumerate(x)]
    #              for x in train_x]
    # serialize([train_ids, train_y], os.path.join(data_dir, 'train_ids.bin'))
    #
    # # dev_x = remove_lcs(dev_x)
    # dev_ids = [[vectorize_sent(q, word2index, oov_embed_size, i, 50) for i, q in enumerate(x)]
    #            for x in dev_x]
    # serialize([dev_ids, dev_y], os.path.join(data_dir, 'dev_ids.bin'))

    test_x = deserialize(os.path.join(data_dir, 'test.bin'))
    # test_x = remove_lcs(test_x)
    test_ids = [[vectorize_sent(q, word2index, oov_embed_size, i, 50) for i, q in enumerate(x)]
                for x in test_x]
    serialize(test_ids, os.path.join(data_dir, 'test_ids.bin'))


if __name__ == '__main__':
    main()
