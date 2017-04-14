# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# longest common sequence match


def get_word_inv(seq):
    word_inv_dict = {}
    for i in range(len(seq) - 1, -1, -1):
        word = seq[i]
        word_inv_dict.setdefault(word, [])
        word_inv_dict[word].append(i)
    return word_inv_dict


def find_max_lis(seq_pos):
    pos_list = []
    for x in seq_pos:
        bound = binary_search(pos_list, x)
        if bound == -1:
            pos_list.append(x)
        else:
            pos_list[bound] = x
    return len(pos_list)


def binary_search(pos_list, key):
    low = 0
    high = len(pos_list) - 1
    if len(pos_list) == 0 or pos_list[high] < key:
        return -1
    while low < high:
        mid = (low + high) // 2
        if pos_list[mid] < key:
            low = mid + 1
        else:
            high = mid
    return high


def get_lcs_ratio(seq1, seq2):
    if len(seq1) > len(seq2):
        long_seq, short_seq = seq1, seq2
    else:
        long_seq, short_seq = seq2, seq1
    word_inv_dict = get_word_inv(short_seq)
    long_seq_pos = []
    for word in long_seq:
        if word in word_inv_dict:
            long_seq_pos.extend(word_inv_dict[word])
    max_lis = find_max_lis(long_seq_pos)
    average_len = (len(seq1) + len(seq2)) / 2
    return max_lis / (average_len + 1e-4)


if __name__ == '__main__':
    a = list('abacd')
    b = list('cabbab')
    print(get_lcs_ratio(a, b))
