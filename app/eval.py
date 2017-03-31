# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime

import numpy as np
import tensorflow as tf
import csv

from app.decorator import exe_time
from preprocess.file_utils import deserialize

# Eval Parameters
tf.flags.DEFINE_string('data_dir', os.path.join('..', 'dataset'), 'Directory containing dataset')

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("checkpoint_dir", os.path.join('..', 'runs',
                                                      '1490760893', 'checkpoints')
                       , "Checkpoint directory from training run")
tf.flags.DEFINE_string('test_file', os.path.join('..', 'dataset', 'test_ids.bin'), '')
tf.flags.DEFINE_string('eval_file', os.path.join('..', 'submit', 'mlstm_pred_6.csv'), '')

FLAGS = tf.flags.FLAGS


@exe_time
def load_data():
    test_ids = deserialize(FLAGS.test_file)
    return test_ids


def cal_dup_score(x):
    score_exp = np.exp(np.asarray(x))
    score_softmax = score_exp / np.reshape(np.sum(score_exp, axis=1), [-1, 1])
    return score_softmax[:, 1]


@exe_time
def test_step(x_batch, sent1, sent2, dropout_keep_prob, logits, sess):
    """
    Evaluates model on a dev set
    """
    sents1 = x_batch[:, 0].tolist()
    sents2 = x_batch[:, 1].tolist()
    feed_dict = {
        sent1: sents1,
        sent2: sents2,
        dropout_keep_prob: 1.0
    }
    scores = sess.run(logits, feed_dict)
    dup_score = cal_dup_score(scores)
    return dup_score


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
        yield data[start_index:end_index], end_index


@exe_time
def write_predictions(all_predictions):
    with open(os.path.join(FLAGS.data_dir, 'sample_submission.csv'), 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
    with open(FLAGS.eval_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i, pre in enumerate(all_predictions):
            prob = '{:.3f}'.format(pre)
            f_csv.writerow([i, prob])
            if i % 10000 == 0:
                print('Writing {} lines...'.format(i))


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

            input1 = graph.get_operation_by_name('sent1').outputs[0]
            input2 = graph.get_operation_by_name('sent2').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            logits = graph.get_operation_by_name('fully_connect/add').outputs[0]

            all_predictions = np.array([])
            for x, end_index in batch_data(test_ids):
                batch_scores = test_step(x, input1, input2, dropout_keep_prob, logits, sess)
                all_predictions = np.concatenate([all_predictions, batch_scores])
                print('Predicting {} lines...'.format(end_index))

            write_predictions(all_predictions)


if __name__ == '__main__':
    tf.app.run()
