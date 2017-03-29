# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib


class MatchLstm:
    def __init__(self, vocab_size, sentence_size, embedding_size,
                 word_embedding, initializer=tf.truncated_normal_initializer(stddev=0.1),
                 num_class=2, window_size=4, name='MatchLstm', initial_lr=1e-3):
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._we = word_embedding
        self._initializer = initializer
        self._name = name
        self._num_class = num_class
        self._window_size = window_size
        self._initial_lr = initial_lr

        self._build_inputs_and_vars()

        self._inference()

        self._initial_optimizer()

    def _build_inputs_and_vars(self):
        self.sent1 = tf.placeholder(shape=[None, self._sentence_size], dtype=tf.int32,
                                    name='sent1')
        self.sent2 = tf.placeholder(shape=[None, self._sentence_size], dtype=tf.int32,
                                    name='sent2')
        self.labels = tf.placeholder(shape=[None, self._num_class], dtype=tf.float32,
                                     name='labels')
        self.dropout_keep_prob = tf.placeholder(shape=[], dtype=tf.float32,
                                                name='dropout_keep_prob')

        self._batch_size = tf.shape(self.sent1)[0]

        self.lr = tf.get_variable(shape=[], dtype=tf.float32, trainable=False,
                                  initializer=tf.constant_initializer(self._initial_lr), name='lr')
        self.new_lr = tf.placeholder(shape=[], dtype=tf.float32,
                                     name='new_lr')
        self.lr_update_op = tf.assign(self.lr, self.new_lr)

        with tf.variable_scope('embedding'):
            self._word_embedding = tf.get_variable(name='word_embedding',
                                                   shape=[self._vocab_size, self._embedding_size],
                                                   initializer=tf.constant_initializer(self._we),
                                                   trainable=False)

        self._embed_pre = self._embed_inputs(self.sent1, self._word_embedding, 'embed_pre')
        self._embed_hyp = self._embed_inputs(self.sent2, self._word_embedding, 'embed_hyp')

    def _inference(self):
        with tf.variable_scope('lstm_s'):
            lstm_s = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size, forget_bias=0.0)
            pre_length = self._length(self.sent1)
            h_s, _ = tf.nn.dynamic_rnn(lstm_s, self._embed_pre, sequence_length=pre_length,
                                       dtype=tf.float32)
            self.h_s = h_s

        with tf.variable_scope('lstm_t'):
            lstm_t = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size, forget_bias=0.0)
            hyp_length = self._length(self.sent2)
            h_t, _ = tf.nn.dynamic_rnn(lstm_t, self._embed_hyp, sequence_length=hyp_length,
                                       dtype=tf.float32)
            self.h_t = h_t

        with tf.name_scope('match_sents'):
            self.lstm_m = contrib.rnn.BasicLSTMCell(num_units=self._embedding_size,
                                                    forget_bias=0.0)
            h_m_arr = tf.TensorArray(dtype=tf.float32, size=self._batch_size)

            i = tf.constant(0)
            c = lambda x, y: tf.less(x, self._batch_size)
            b = lambda x, y: self._match_sent(x, y)
            res = tf.while_loop(cond=c, body=b, loop_vars=(i, h_m_arr))
            self.h_m_tensor = tf.squeeze(res[-1].stack(), axis=[1])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_m_tensor, self.dropout_keep_prob)

        with tf.variable_scope('fully_connect'):
            w1 = tf.get_variable(shape=[self._embedding_size, self._num_class],
                                 initializer=contrib.layers.xavier_initializer(),
                                 name='w1')
            b1 = tf.Variable(tf.constant(0.1, shape=[self._num_class]), name="b1")
            self.logits = tf.matmul(self.h_drop, w1) + b1

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                    logits=self.logits,
                                                                    name='cross_entropy')
            self.loss_op = tf.reduce_mean(cross_entropy, name='loss_op')

        # Accuracy
        with tf.name_scope("accuracy"):
            self.predict_op = tf.arg_max(self.logits, dimension=1, name='predict_op')
            hard_labels = tf.argmax(self.labels, axis=1, name='hard_labels')
            correct_predictions = tf.cast(tf.equal(self.predict_op, hard_labels), tf.float32)
            self.accuracy_op = tf.reduce_mean(correct_predictions,
                                              name='accuracy_op')

            hard_labels = tf.reshape(tf.cast(hard_labels, dtype=tf.float32), [-1, 1])
            positive_cnt = tf.reduce_sum(hard_labels)
            correct_predictions = tf.reshape(correct_predictions, [1, -1])
            positive_true_cnt = tf.reshape(tf.matmul(correct_predictions, hard_labels), [])
            self.recall_op = positive_true_cnt / (positive_cnt + 1e-3)

    def _match_sent(self, i, h_m_arr):
        h_s_i = self.h_s[i]
        h_t_i = self.h_t[i]
        length_s_i = self._length(self.sent1[i])
        length_t_i = self._length(self.sent2[i])

        state = self.lstm_m.zero_state(batch_size=1, dtype=tf.float32)

        k = tf.constant(0)
        c = lambda a, x, y, z, s: tf.less(a, length_t_i)
        b = lambda a, x, y, z, s: self._match_attention(a, x, y, z, s)
        res = tf.while_loop(cond=c, body=b, loop_vars=(k, h_s_i, h_t_i, length_s_i, state))

        final_state_h = res[-1].h
        h_m_arr = h_m_arr.write(i, final_state_h)

        i = tf.add(i, 1)
        return i, h_m_arr

    def _match_attention(self, k, h_s, h_t, length_s, state):
        h_t_k = tf.reshape(h_t[k], [1, -1])
        h_s_j = tf.slice(h_s, begin=[0, 0], size=[length_s, self._embedding_size])

        with tf.variable_scope('attention_w'):
            w_s = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_s')
            w_t = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_t')
            w_m = tf.get_variable(shape=[self._embedding_size, self._embedding_size],
                                  initializer=self._initializer, name='w_m')
            w_e = tf.get_variable(shape=[self._embedding_size, 1],
                                  initializer=self._initializer, name='w_e')

        with tf.variable_scope('align'):
            last_m_h = state.h
            sum_h = tf.matmul(h_s_j, w_s) + tf.matmul(h_t_k, w_t) + tf.matmul(last_m_h, w_m)
            e_kj = tf.matmul(tf.tanh(sum_h), w_e)
            a_kj = tf.nn.softmax(e_kj)
            alpha_k = tf.matmul(a_kj, h_s_j, transpose_a=True)
            alpha_k.set_shape([1, self._embedding_size])
            m_k = tf.concat([alpha_k, h_t_k], axis=1)

        with tf.variable_scope('lstm_m'):
            _, new_state = self.lstm_m(inputs=m_k, state=state)

        k = tf.add(k, 1)
        return k, h_s, h_t, length_s, new_state

    @staticmethod
    def _embed_inputs(inputs, embeddings, name):
        return tf.nn.embedding_lookup(embeddings, inputs, name)

    @staticmethod
    def _length(sequence):
        mask = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(mask, axis=-1)
        return length

    def _initial_optimizer(self):
        with tf.variable_scope('step'):
            self.global_step = tf.get_variable(shape=[],
                                               initializer=tf.constant_initializer(0),
                                               dtype=tf.int32,
                                               name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
        self.train_op = self._optimizer.minimize(self.loss_op, global_step=self.global_step)


if __name__ == '__main__':
    pass
