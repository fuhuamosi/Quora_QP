# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as contrib


class TextCnn:
    def __init__(self, vocab_size, sentence_size, embedding_size,
                 word_embedding, filter_sizes, num_filters,
                 num_classes=2, initial_lr=1e-3):
        self.sent1 = tf.placeholder(tf.int32, [None, sentence_size], name='sent1')
        self.sent2 = tf.placeholder(tf.int32, [None, sentence_size], name='sent2')
        self.labels = tf.placeholder(tf.float32, [None, num_classes], 'labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], 'dropout_keep_prob')

        with tf.variable_scope('embedding'):
            self.word_embedding = tf.get_variable(name='word_embedding',
                                                  shape=[vocab_size, embedding_size],
                                                  initializer=tf.constant_initializer(
                                                      word_embedding),
                                                  trainable=False)
            embedded_sent1 = tf.nn.embedding_lookup(self.word_embedding, self.sent1)
            embedded_sent2 = tf.nn.embedding_lookup(self.word_embedding, self.sent2)
            self.embedded_expanded_sent1 = tf.expand_dims(embedded_sent1, -1)
            self.embedded_expanded_sent2 = tf.expand_dims(embedded_sent2, -1)

        pooled_outputs_1 = []
        pooled_outputs_2 = []
        for filter_size in filter_sizes:
            with tf.name_scope('conv_maxpool_{}'.format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv1 = tf.nn.conv2d(input=self.embedded_expanded_sent1,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID',
                                     name='conv_1')
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu_1')
                pooled1 = tf.nn.max_pool(h1,
                                         ksize=[1, sentence_size - filter_size + 1, 1, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='VALID',
                                         name='pool_1')
                pooled_outputs_1.append(pooled1)

                conv2 = tf.nn.conv2d(input=self.embedded_expanded_sent2,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID',
                                     name='conv_2')
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu_2')
                pooled2 = tf.nn.max_pool(h2,
                                         ksize=[1, sentence_size - filter_size + 1, 1, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='VALID',
                                         name='pool_2')
                pooled_outputs_2.append(pooled2)

        num_filters_total = len(filter_sizes) * num_filters
        pooled_reshape_1 = tf.reshape(tf.concat(pooled_outputs_1, 3), [-1, num_filters_total])
        pooled_reshape_2 = tf.reshape(tf.concat(pooled_outputs_2, 3), [-1, num_filters_total])

        # pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
        # pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        #
        # pooled_len_1 = tf.sqrt(tf.reduce_sum(pooled_flat_1 * pooled_flat_1, 1))
        # pooled_len_2 = tf.sqrt(tf.reduce_sum(pooled_flat_2 * pooled_flat_2, 1))
        # pooled_mul = tf.reduce_sum(pooled_flat_1 * pooled_flat_2, 1)
        #
        # with tf.name_scope('output'):
        #     self.cos_score = pooled_mul / (pooled_len_1 * pooled_len_2)
        #     self.zero_score = tf.ones_like(self.cos_score) - self.cos_score
        #     self.logits = tf.reshape(tf.concat([self.zero_score, self.cos_score], 0),
        #                              [-1, num_classes])

        with tf.name_scope('loss'):
            pooled_sub = tf.nn.dropout(pooled_reshape_1 - pooled_reshape_2,
                                       self.dropout_keep_prob)
            W2 = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.nn.xw_plus_b(pooled_sub, W2, b2)
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

        with tf.variable_scope('step'):
            self.global_step = tf.get_variable(shape=[],
                                               initializer=tf.constant_initializer(0),
                                               dtype=tf.int32,
                                               name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=initial_lr, beta1=0.9, beta2=0.999)
        self.train_op = self._optimizer.minimize(self.loss_op, global_step=self.global_step)


if __name__ == '__main__':
    pass
