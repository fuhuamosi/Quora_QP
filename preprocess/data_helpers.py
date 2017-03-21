import numpy as np
import re
import csv
from collections import namedtuple
from typing import Dict
import os
from preprocess.file_utils import deserialize, serialize

PAD_ID = 0
NULL_ID = 1
data_dir = os.path.join('..', 'dataset')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"'s", " 's", string)
    string = re.sub(r"'ve", " 've", string)
    string = re.sub(r"n't", " n't", string)
    string = re.sub(r"'re", " 're", string)
    string = re.sub(r"'d", " 'd", string)
    string = re.sub(r"'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
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


def vectorize_sent(sent, word2idx: Dict, oov_size, sent_size=50):
    trim_sent = sent[:sent_size]
    dict_len = len(word2idx)
    sent_ids = [word2idx.get(w, hash(w) % oov_size + dict_len) for w in trim_sent]
    pad_length = sent_size - len(sent_ids)
    sent_ids.extend([PAD_ID] * pad_length)
    return sent_ids


def vectorize_y(ys, num_class):
    y_dis = np.zeros([len(ys), num_class], dtype=np.float32)
    for i in range(len(ys)):
        y_dis[i, ys[i]] = 1.0
    return y_dis.tolist()


def sample_eval_data(dev_x, dev_y, size):
    x_arr = np.array(dev_x)
    y_arr = np.array(dev_y)
    shuffle_indices = np.random.permutation(len(dev_x))
    return x_arr[shuffle_indices[:size]], y_arr[shuffle_indices[:size]]


if __name__ == '__main__':
    # train_x, train_y = deserialize(os.path.join(data_dir, 'train.bin'))
    # dev_x, dev_y = deserialize(os.path.join(data_dir, 'dev.bin'))
    # test_x = deserialize(os.path.join(data_dir, 'test.bin'))
    #
    # word2index = deserialize(os.path.join(data_dir, 'word2index_glove.bin'))
    # word_embeddings = deserialize(os.path.join(data_dir, 'word_embeddings_glove.bin'))
    # oov_embed_size = len(word_embeddings) - len(word2index)
    #
    # train_ids = [[vectorize_sent(q, word2index, oov_embed_size) for q in x] for x in train_x]
    # serialize([train_ids, train_y], os.path.join(data_dir, 'train_ids.bin'))
    #
    # dev_ids = [[vectorize_sent(q, word2index, oov_embed_size) for q in x] for x in dev_x]
    # serialize([dev_ids, dev_y], os.path.join(data_dir, 'dev_ids.bin'))
    #
    # test_ids = [[vectorize_sent(q, word2index, oov_embed_size) for q in x] for x in test_x]
    # serialize(test_ids, os.path.join(data_dir, 'test_ids.bin'))
    pass
