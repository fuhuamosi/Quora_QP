# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import csv
from preprocess.file_utils import deserialize
from app.decorator import exe_time

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", os.path.join('..', 'dataset', 'runs',
                                                      '1490166291', 'checkpoints')
                       , "Checkpoint directory from training run")
tf.flags.DEFINE_string('data_dir', os.path.join('..', 'dataset'), 'Directory containing dataset')

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


@exe_time
def load_data():
    test_ids, _ = deserialize(os.path.join(FLAGS.data_dir, 'dev_ids.bin'))
    return test_ids


def cal_dup_score(x):
    score_exp = np.exp(np.asarray(x))
    score_softmax = score_exp / np.reshape(np.sum(score_exp, axis=1), [-1, 1])
    return score_softmax[:, 1]


@exe_time
def test_step(x_batch, pre, hyp, dropout_prob, logits, sess):
    """
    Evaluates model on a dev set
    """
    sents1 = x_batch[:, 0].tolist()
    sents2 = x_batch[:, 1].tolist()
    feed_dict = {
        pre: sents1,
        hyp: sents2,
        dropout_prob: 1.0
    }
    scores = sess.run(logits, feed_dict)
    dup_score = cal_dup_score(scores)
    print(dup_score)


def batch_data(data, batch_size=1000):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


def main(_):
    test_ids = load_data()

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input1 = graph.get_operation_by_name('premises').outputs[0]
            input2 = graph.get_operation_by_name('hypotheses').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            logits = graph.get_operation_by_name('MatchLstm_fully_connect/add').outputs[0]

            for x in batch_data(test_ids):
                test_step(x, input1, input2, dropout_keep_prob, logits, sess)


if __name__ == '__main__':
    tf.app.run()
