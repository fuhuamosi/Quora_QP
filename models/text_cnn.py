# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib


class TextCnn:
    def __init__(self, vocab_size, sentence_size, embedding_size,
                 word_embedding, initializer=tf.truncated_normal_initializer(stddev=0.1),
                 num_class=3, window_size=4, name='MatchLstm', initial_lr=1e-3):
        pass
