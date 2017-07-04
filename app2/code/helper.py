# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def filter_test_data(gen1, gen2, labels, test1, test2):
    test_set = set([tuple(x) for x in test1]) | set([tuple(x) for x in test2])
    new_gen1, new_gen2, new_labels = np.array([]), np.array([]), np.array([])
    for i in range(len(gen1)):
        s1 = tuple(gen1[i])
        s2 = tuple(gen2[i])
        if s1 not in test_set and s2 not in test_set:
            np.append(new_gen1, list(s1))
            np.append(new_gen2, list(s2))
            np.append(new_labels, labels[i])
    return new_gen1, new_gen2, new_labels
