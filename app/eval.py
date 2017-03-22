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
from preprocess.data_helpers import vectorize_y, batch_iter, sample_eval_data
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
    test_ids = deserialize(os.path.join(FLAGS.data_dir, 'test_ids.bin'))
    return test_ids


def cal_dup_score(scores):
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores)
    return exp_scores[0, 1] / exp_scores_sum


@exe_time
def test_step(x, pre, hyp, dropout_prob, logits, sess):
    """
    Evaluates model on a dev set
    """
    sents1 = np.reshape(np.array(x[0]), [1, -1])
    sents2 = np.reshape(np.array(x[1]), [1, -1])
    feed_dict = {
        pre: sents1,
        hyp: sents2,
        dropout_prob: 1.0
    }
    scores = sess.run(logits, feed_dict)
    dup_score = cal_dup_score(scores)
    dup_score = '{:.4f}'.format(dup_score)
    print(dup_score)


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

            for x in test_ids:
                test_step(x, input1, input2, dropout_keep_prob, logits, sess)
                break


if __name__ == '__main__':
    tf.app.run()
