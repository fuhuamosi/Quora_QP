import numpy as np
import re
import csv
from collections import namedtuple
from typing import Dict
import os
from preprocess.file_utils import deserialize, serialize
from preprocess.lcs_match import get_lcs_ratio
import math
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


def remove_common_words(sents1, sents2):
    new_sents1, new_sents2 = [], []
    length = len(sents1[0])
    for s1, s2 in zip(sents1, sents2):
        new_s1 = []
        new_s2 = []
        common_words = set(s1) & set(s2)

        for w in s1:
            if w not in common_words:
                new_s1.append(w)
        for w in s2:
            if w not in common_words:
                new_s2.append(w)

        if len(new_s1) == 0:
            new_s1.append(NULL_ID)
        if len(new_s2) == 0:
            new_s2.append(NULL_ID)

        new_s1.extend([PAD_ID] * (length - len(new_s1)))
        new_s2.extend([PAD_ID] * (length - len(new_s2)))

        new_sents1.append(new_s1)
        new_sents2.append(new_s2)
    return new_sents1, new_sents2


def remove_rare_words(sents1, sents2, max_id):
    new_sents1, new_sents2 = [], []
    length = len(sents1[0])
    for s1, s2 in zip(sents1, sents2):
        new_s1 = []
        new_s2 = []

        for w in s1:
            if w < max_id:
                new_s1.append(w)
        for w in s2:
            if w < max_id:
                new_s2.append(w)

        if len(new_s1) == 0:
            new_s1.append(NULL_ID)
        if len(new_s2) == 0:
            new_s2.append(NULL_ID)

        new_s1.extend([PAD_ID] * (length - len(new_s1)))
        new_s2.extend([PAD_ID] * (length - len(new_s2)))

        new_sents1.append(new_s1)
        new_sents2.append(new_s2)
    return new_sents1, new_sents2


def unpack_x_batch(x_batch):
    sents1, sents2 = [], []
    for s in x_batch:
        sents1.append(s[0])
        sents2.append(s[1])
    return sents1, sents2


def get_idf_ratio(a, b, idf_dict):
    common_ids = set(a) & set(b)
    common_weights = sum([idf_dict.get(x, 0) for x in common_ids])
    all_words = set(a) | set(b)
    all_weights = sum([idf_dict.get(x, 0) for x in all_words])
    return common_weights / (all_weights + 1e-4)


def get_levenshtein_ratio(a, b):
    len1 = len(a)
    len2 = len(b)
    min_dis = np.zeros(shape=[len1 + 1, len2 + 1], dtype=np.int32)
    for i in range(1, len1 + 1):
        min_dis[i][0] = i
    for i in range(1, len2 + 1):
        min_dis[0][i] = i
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if a[i - 1] != b[j - 1]:
                min_dis[i][j] = min(min_dis[i - 1][j - 1], min_dis[i - 1][j], min_dis[i][j - 1]) + 1
            else:
                min_dis[i][j] = min_dis[i - 1][j - 1]
    return min_dis[len1][len2] / ((len1 + len2) / 2)


def get_move_ratio(a, b, word_embeddings):
    def get_move(x, y):
        all_dis = 0
        for w1 in x:
            embed_w1 = word_embeddings[w1]
            min_dis = 1000.0
            for w2 in y:
                embed_w2 = word_embeddings[w2]
                min_dis = min(min_dis, np.linalg.norm(embed_w1 - embed_w2))
            all_dis += min_dis
        return all_dis / (len(x) + 1e-4) / 10

    dis1 = get_move(a, b)
    dis2 = get_move(b, a)
    average_dis = (dis1 + dis2) / 2
    return average_dis


def get_jaccard_ratio(a, b):
    common_ids = set(a) & set(b)
    all_words = set(a) | set(b)
    return len(common_ids) / (len(all_words) + 1e-4)


def get_two_gram_ratio(a, b, idf_dict):
    gram1 = []
    for i in range(len(a) - 1):
        gram1.append(str(a[i]) + ' ' + str(a[i + 1]))
    gram2 = []
    for i in range(len(b) - 1):
        gram2.append(str(b[i]) + ' ' + str(b[i + 1]))
    gram1 = set(gram1)
    gram2 = set(gram2)
    commons = gram1 & gram2
    common_weight = sum([sum([idf_dict.get(int(z), 0) for z in x.split(' ')]) for x in commons])
    alls = gram1 | gram2
    all_weight = sum([sum([idf_dict.get(int(z), 0) for z in x.split(' ')]) for x in alls])
    return common_weight / (all_weight + 1e-4)


def get_freq_ratio(a, b, question_freq: dict, max_freq=161):
    a = [str(x) for x in a]
    a = ' '.join(a)
    b = [str(x) for x in b]
    b = ' '.join(b)
    a_freq = question_freq.get(a, 0)
    b_freq = question_freq.get(b, 0)
    return max(a_freq, b_freq) / max_freq


def get_inter_radios(a, b, inter_dict, max_cnt=30):
    q1 = ' '.join([str(x) for x in a])
    q2 = ' '.join([str(x) for x in b])
    inter_cnt = len(inter_dict.get(q1, set()) & inter_dict.get(q2, set()))
    return inter_cnt / max_cnt


def get_extra_features(sents1, sents2, idf_dict, word_embeddings, question_freq, inter_dict):
    s1 = [list(filter(lambda x: x != PAD_ID, s)) for s in sents1]
    s2 = [list(filter(lambda x: x != PAD_ID, s)) for s in sents2]
    idf_ratios = [get_idf_ratio(a, b, idf_dict) for a, b in zip(s1, s2)]
    freq_ratios = [get_freq_ratio(a, b, question_freq) for a, b in zip(s1, s2)]
    two_gram_ratios = [get_two_gram_ratio(a, b, idf_dict) for a, b in zip(s1, s2)]
    inter_radios = [get_inter_radios(a, b, inter_dict) for a, b in zip(s1, s2)]
    # lcs_ratios = [get_lcs_ratio(a, b) for a, b in zip(s1, s2)]
    # jaccard_ratios = [get_jaccard_ratio(a, b) for a, b in zip(s1, s2)]
    # levenshtein_ratios = [get_levenshtein_ratio(a, b) for a, b in zip(s1, s2)]
    # move_ratios = [get_move_ratio(a, b, word_embeddings) for a, b in zip(s1, s2)]

    extra_features = [[a, b, c, d] for a, b, c, d in
                      zip(idf_ratios, freq_ratios, two_gram_ratios, inter_radios)]
    return np.array(extra_features)


def get_idf_dict(text_list, min_cnt=5):
    idf_dict = {}
    for x in text_list:
        x_set = set()
        for w in x:
            if w == 0:
                break
            if w not in x_set:
                idf_dict.setdefault(w, 0)
                idf_dict[w] += 1
                x_set.add(w)
    for w in idf_dict:
        cnt = idf_dict[w]
        if cnt < 5:
            idf_dict[w] = 0
        else:
            idf_dict[w] = math.log(len(text_list) / idf_dict[w])
    return idf_dict


def get_question_freq(text_list):
    freq_dict = {}
    for t in text_list:
        q = [str(x) for x in t]
        q_str = ' '.join(q)
        freq_dict.setdefault(q_str, 0)
        freq_dict[q_str] += 1
    # freq_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    return freq_dict


def get_iter_dict(seqs1, seqs2):
    iter_dict = {}
    assert len(seqs1) == len(seqs2)
    for i in range(len(seqs1)):
        q1 = ' '.join([str(x) for x in seqs1[i]])
        q2 = ' '.join([str(x) for x in seqs2[i]])
        iter_dict.setdefault(q1, set())
        iter_dict.setdefault(q2, set())
        iter_dict[q1].add(q2)
        iter_dict[q2].add(q1)
    return iter_dict


def main():
    train_x, train_y = deserialize(os.path.join(data_dir, 'train.bin'))
    dev_x, dev_y = deserialize(os.path.join(data_dir, 'dev.bin'))

    word2index = deserialize(os.path.join(data_dir, 'word2index_word2vec.bin'))
    word_embeddings = deserialize(os.path.join(data_dir, 'word_embeddings_word2vec.bin'))
    oov_embed_size = len(word_embeddings) - len(word2index)

    train_ids = [[vectorize_sent(q, word2index, oov_embed_size, i, 50) for i, q in enumerate(x)]
                 for x in train_x]
    serialize([train_ids, train_y], os.path.join(data_dir, 'train_ids.bin'))

    dev_ids = [[vectorize_sent(q, word2index, oov_embed_size, i, 50) for i, q in enumerate(x)]
               for x in dev_x]
    serialize([dev_ids, dev_y], os.path.join(data_dir, 'dev_ids.bin'))

    test_x = deserialize(os.path.join(data_dir, 'test.bin'))
    test_ids = [[vectorize_sent(q, word2index, oov_embed_size, i, 50) for i, q in enumerate(x)]
                for x in test_x]
    serialize(test_ids, os.path.join(data_dir, 'test_ids.bin'))


if __name__ == '__main__':
    main()
