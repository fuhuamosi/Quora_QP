# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from tensorflow.python.platform import gfile


def serialize(data, file_path):
    if gfile.Exists(file_path):
        print('{} already exists'.format(file_path))
        return
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        print('saving {} lines to {}'.format(len(data), file_path))


def deserialize(file_path):
    if not gfile.Exists(file_path):
        raise RuntimeError('{} does not exist'.format(file_path))
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print('loading {} lines from {}'.format(len(data), file_path))
    return data
