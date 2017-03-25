import numpy as np
import re
import csv
from collections import namedtuple
from typing import Dict
import os
from preprocess.file_utils import deserialize, serialize
from preprocess.lcs_match import get_lcs_ratio

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
        y_dis[i, 1] = float(ys[i])
        y_dis[i, 0] = 1.0 - y_dis[i, 1]
    return y_dis.tolist()


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


def transform_x(x):
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


def main():
    train_x, train_y = deserialize(os.path.join(data_dir, 'train.bin'))
    dev_x, dev_y = deserialize(os.path.join(data_dir, 'dev.bin'))

    word2index = deserialize(os.path.join(data_dir, 'word2index_glove.bin'))
    word_embeddings = deserialize(os.path.join(data_dir, 'word_embeddings_glove.bin'))
    oov_embed_size = len(word_embeddings) - len(word2index)

    alpha = 0.7

    train_y = transform_y(train_x, train_y, alpha)
    train_x = transform_x(train_x)
    train_ids = [[vectorize_sent(q, word2index, oov_embed_size) for q in x] for x in train_x]
    serialize([train_ids, train_y], os.path.join(data_dir, 'train_ids_{}_2.bin'.format(alpha)))

    dev_y = transform_y(dev_x, dev_y, alpha)
    dev_x = transform_x(dev_x)
    dev_ids = [[vectorize_sent(q, word2index, oov_embed_size) for q in x] for x in dev_x]
    serialize([dev_ids, dev_y], os.path.join(data_dir, 'dev_ids_{}_2.bin'.format(alpha)))

    test_x = deserialize(os.path.join(data_dir, 'test.bin'))
    test_x = transform_x(test_x)
    test_ids = [[vectorize_sent(q, word2index, oov_embed_size) for q in x] for x in test_x]
    serialize(test_ids, os.path.join(data_dir, 'test_ids_{}_2.bin'.format(alpha)))


if __name__ == '__main__':
    main()
