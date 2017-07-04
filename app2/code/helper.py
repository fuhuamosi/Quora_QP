# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

f = None
rank = None


def generate_samples(train_file, idx_train):
    train_data = pd.read_csv(train_file)
    train_data = train_data.iloc[idx_train, :]

    qid_question_dict = {}
    duplicates_set = set()
    for index, row in train_data.iterrows():
        qid_question_dict[row['qid1']] = row['question1']
        qid_question_dict[row['qid2']] = row['question2']
        duplicates_set.add((row['qid1'], row['qid2']))
        duplicates_set.add((row['qid2'], row['qid1']))
    duplicate_questions = train_data[train_data.is_duplicate == 1]

    f = dict.fromkeys(qid_question_dict)
    rank = dict.fromkeys(qid_question_dict)

    for x in f:
        f[x] = x
        rank[x] = 0

    for index, row in duplicate_questions.iterrows():
        make(row['qid1'], row['qid2'])

    for x in f:
        find(x)

    duplicates = list(f.items())
    duplicates = sorted(duplicates, key=lambda aa: aa[1])
    duplicates_dict = dict.fromkeys(f)
    for d in duplicates_dict:
        duplicates_dict[d] = set()
    for d in duplicates:
        duplicates_dict[d[1]].add(d[0])
    new_duplicates = []
    for d in duplicates_dict:
        temp = list(duplicates_dict[d])
        for a in range(len(temp)):
            x = temp[a]
            for b in range(a + 1, len(temp)):
                y = temp[b]
                if (x, y) not in duplicates_set and (y, x) not in duplicates_set:
                    new_duplicates.append((x, y))
                else:
                    pass
    text0 = [qid_question_dict[x[0]] for x in new_duplicates]
    text1 = [qid_question_dict[x[1]] for x in new_duplicates]
    labels = np.ones(len(text0))
    return text0, text1, labels


def find(a):
    if f[a] != a:
        f[a] = find(f[a])
    return f[a]


def make(a, b):
    a = find(a)
    b = find(b)
    if a == b: return
    if rank[a] > rank[b]:
        f[b] = a
    else:
        if rank[a] == rank[b]:
            rank[b] += 1
        f[a] = b
