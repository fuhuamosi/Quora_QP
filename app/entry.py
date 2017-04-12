# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import time

import tensorflow as tf

from app.decorator import exe_time
from models.match_lstm import MatchLstm
from models.text_cnn import TextCnn
from preprocess.data_helpers import batch_iter, sample_eval_data, \
    unpack_x_batch, get_extra_features, remove_rare_words
from preprocess.file_utils import deserialize

tf.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate')
tf.flags.DEFINE_float('decay_ratio', 0.95, 'Learning rate decay ratio')
tf.flags.DEFINE_float('max_grad_norm', 40.0, 'Clip gradients to this norm.')
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
tf.flags.DEFINE_integer('sent_size', 50, 'Max sentence size.')
tf.flags.DEFINE_integer('num_class', 2, 'Max sentence size.')
tf.flags.DEFINE_integer('num_epochs', 3, 'Number of epochs to train for.')
tf.flags.DEFINE_integer('embedding_size', 300, 'Embedding size for embedding matrices.')
tf.flags.DEFINE_string('data_dir', os.path.join('..', 'dataset'), 'Directory containing dataset')

tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("eval_size", 1000, "Evaluate model on dev set's size")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_string('train_file', os.path.join('..', 'dataset', 'train_ids.bin'), '')
tf.flags.DEFINE_string('dev_file', os.path.join('..', 'dataset', 'dev_ids.bin'), '')
tf.flags.DEFINE_string('idf_file', os.path.join('..', 'dataset', 'idf_dict.bin'), '')
tf.flags.DEFINE_string('embed_file', os.path.join('..', 'dataset', 'word_embeddings_word2vec2.bin'), '')
tf.flags.DEFINE_string('word2inx_file', os.path.join('..', 'dataset', 'word2index_word2vec.bin'), '')

FLAGS = tf.flags.FLAGS

idf_dict = deserialize(FLAGS.idf_file)
max_id = len(deserialize(FLAGS.word2inx_file))


@exe_time
def load_data():
    train_ids, train_y = deserialize(FLAGS.train_file)
    dev_ids, dev_y = deserialize(FLAGS.dev_file)
    return train_ids, train_y, dev_ids, dev_y


@exe_time
def load_embed():
    word_embeddings = deserialize(FLAGS.embed_file)
    return word_embeddings


@exe_time
def train_step(x_batch, y_batch, train_summary_op,
               train_summary_writer, model, sess):
    """
    A single training step
    """
    sents1, sents2 = unpack_x_batch(x_batch)
    extra_features = get_extra_features(sents1, sents2, idf_dict)
    sents1, sents2 = remove_rare_words(sents1, sents2, max_id)
    feed_dict = {
        model.sent1: sents1,
        model.sent2: sents2,
        model.labels: y_batch,
        model.dropout_keep_prob: FLAGS.dropout_keep_prob,
        model.extra_features: extra_features
    }
    _, step, summaries, loss, accuracy, recall = sess.run(
        [model.train_op, model.global_step, train_summary_op,
         model.loss_op, model.accuracy_op, model.recall_op],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:.4f}, rec {:.4f}, acc {:.4f}".format(time_str, step,
                                                                    loss, recall, accuracy))
    train_summary_writer.add_summary(summaries, step)


@exe_time
def dev_step(x_batch, y_batch, dev_summary_op,
             dev_summary_writer, model, sess):
    """
    Evaluates model on a dev set
    """
    sents1, sents2 = unpack_x_batch(x_batch)
    extra_features = get_extra_features(sents1, sents2, idf_dict)
    sents1, sents2 = remove_rare_words(sents1, sents2, max_id)
    feed_dict = {
        model.sent1: sents1,
        model.sent2: sents2,
        model.labels: y_batch,
        model.dropout_keep_prob: 1.0,
        model.extra_features: extra_features
    }
    step, summaries, loss, accuracy, recall = sess.run(
        [model.global_step, dev_summary_op,
         model.loss_op, model.accuracy_op, model.recall_op],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:.4f}, rec {:.4f}, acc {:.4f}".format(time_str, step,
                                                                    loss, recall, accuracy))
    dev_summary_writer.add_summary(summaries, step)


@exe_time
def get_model(word_embeddings, model_name):
    model = None
    if model_name == 'match_lstm':
        model = MatchLstm(vocab_size=len(word_embeddings),
                          sentence_size=FLAGS.sent_size,
                          embedding_size=FLAGS.embedding_size,
                          word_embedding=word_embeddings,
                          initial_lr=FLAGS.learning_rate,
                          num_class=FLAGS.num_class,
                          extra_cnt=3)
    elif model_name == 'text_cnn':
        model = TextCnn(vocab_size=len(word_embeddings),
                        sentence_size=FLAGS.sent_size,
                        embedding_size=FLAGS.embedding_size,
                        word_embedding=word_embeddings,
                        initial_lr=FLAGS.learning_rate,
                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                        num_filters=FLAGS.num_filters,
                        num_classes=FLAGS.num_class)
    return model


def main(_):
    train_ids, train_y, dev_ids, dev_y = load_data()
    word_embeddings = load_embed()

    # Training

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            model = get_model(word_embeddings, 'match_lstm')

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join('..', 'runs', timestamp))
            print('Writing to {}\n'.format(out_dir))

            loss_summary = tf.summary.scalar(name='loss', tensor=model.loss_op)
            accuracy_summary = tf.summary.scalar(name='accuracy', tensor=model.accuracy_op)
            recall_summary = tf.summary.scalar(name='recall', tensor=model.recall_op)

            train_summary_op = tf.summary.merge(
                [loss_summary, accuracy_summary, recall_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([loss_summary, accuracy_summary, recall_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(var_list=tf.global_variables(),
                                   max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer())

            batches = batch_iter(list(zip(train_ids, train_y)), FLAGS.batch_size,
                                 FLAGS.num_epochs)

            for batch in batches:
                x_batch, y_batch = batch[:, 0], batch[:, 1]
                train_step(x_batch, y_batch, train_summary_op,
                           train_summary_writer, model, sess)
                current_step = tf.train.global_step(sess, model.global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print('\nEvaluation:')
                    x_dev, y_dev = sample_eval_data(dev_ids, dev_y, FLAGS.eval_size)
                    dev_step(x_dev, y_dev, dev_summary_op,
                             dev_summary_writer, model, sess)
                    print('')
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, save_path=checkpoint_prefix,
                                      global_step=model.global_step)
                    print('Saved model checkpoint to {}\n'.format(path))


if __name__ == '__main__':
    tf.app.run()
