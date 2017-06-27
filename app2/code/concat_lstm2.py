# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import codecs
import csv
import os
import re

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, \
    Conv2D, MaxPool2D, Reshape
from keras.layers.merge import concatenate, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from preprocess.data_helpers import get_idf_dict, get_extra_features, \
    get_question_freq, get_inter_dict

"""
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!
"""

np.random.seed(888)

########################################
## set directories and parameters
########################################
BASE_DIR = os.path.join('..', 'input')
EMBEDDING_FILE = os.path.join(BASE_DIR, 'GoogleNews-vectors-negative300.bin')
TRAIN_DATA_FILE = os.path.join(BASE_DIR, 'train.csv')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'test.csv')
GENERATE_DATA_FILE = os.path.join(BASE_DIR, 'generate_data.csv')
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 250
num_dense = 200
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

class0_weight = 1.382
class1_weight = 0.680

max_cnt = 10000000

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_{:d}_{:d}_{:.2f}_{:.2f}'.format(num_lstm, num_dense,
                                              rate_drop_lstm, rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,
                                             binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
########################################
print('Processing text dataset')


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_word_list(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text


cnt = 0
texts_1 = []
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_word_list(values[3]))
        texts_2.append(text_to_word_list(values[4]))
        labels.append(int(values[5]))
        if cnt > max_cnt:
            break
        cnt += 1
print('Found %s texts in train.csv' % len(texts_1))

cnt = 0
texts_3 = []
texts_4 = []
labels2 = []
with codecs.open(GENERATE_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for values in reader:
        texts_3.append(text_to_word_list(values[3]))
        texts_4.append(text_to_word_list(values[4]))
        labels2.append(int(values[5]))
        if cnt > max_cnt:
            break
        cnt += 1
print('Found %s texts in generate_data.csv' % len(texts_3))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
sequences_3 = tokenizer.texts_to_sequences(texts_3)
sequences_4 = tokenizer.texts_to_sequences(texts_4)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
data_3 = pad_sequences(sequences_3, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
data_4 = pad_sequences(sequences_4, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train], data_3, data_4))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train], data_4, data_3))
labels_train = np.concatenate((labels[idx_train], labels[idx_train], labels2, labels2))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= class1_weight
    weight_val[labels_val == 0] = class0_weight

# all_sequences = sequences_1 + sequences_2 + test_sequences_1 + test_sequences_2
# idf_dict = get_idf_dict(all_sequences)
# question_freq = get_question_freq(all_sequences)
# inter_dict = get_inter_dict(sequences_1 + test_sequences_1, sequences_2 + test_sequences_2)
#
# train_features = get_extra_features(data_1_train.tolist(), data_2_train.tolist(), idf_dict,
#                                     embedding_matrix, question_freq, inter_dict)
# val_features = get_extra_features(data_1_val.tolist(), data_2_val.tolist(), idf_dict,
#                                   embedding_matrix, question_freq, inter_dict)
# test_features = get_extra_features(test_data_1.tolist(), test_data_2.tolist(), idf_dict,
#                                    embedding_matrix, question_freq, inter_dict)
# extra_feature_num = len(train_features[0])

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

add_distance1 = add([x1, y1])
mul_distance1 = multiply([x1, y1])
merged = concatenate([add_distance1, mul_distance1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: class0_weight, 1: class1_weight}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input],
              outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
# model.summary()
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train,
                 validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                 epochs=200, batch_size=2048, shuffle=True,
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

print('best val score: {}'.format(bst_val_score))
