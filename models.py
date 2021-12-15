#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:45:21 2021

@author: spiros
"""
import os
import time
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import regularizers

import plot_conf_matrix
from build_model import build_model


def run_rnn_model(data, labels, epochs=20, num_classes=2,
                  timesteps=2, features=1,
                  problem_type='multiclass', seed=0):
    """
    Execute RNN model.

    Parameters
    ----------
    data : np.bdarray
        DESCRIPTION.
    labels : np.ndarray
        DESCRIPTION.
    epochs : int, optional
        DESCRIPTION. The default is 20.
    num_classes : int, optional
        DESCRIPTION. The default is 2.
    timesteps : int, optional
        DESCRIPTION. The default is 2.
    features : int, optional
        DESCRIPTION. The default is 1.
    problem_type : str, optional
        DESCRIPTION. The default is 'multiclass'.
    seed : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    output : dict
        DESCRIPTION
    """
    # ----------------------------------------------------------------------------
    class_names = np.array(['PY', 'SOM', 'PV', 'VIP']).astype(str)
    # ----------------------------------------------------------------------------
    # Split the dataset into train and test sets.
    train_data, test_data, \
        train_labels, test_labels = train_test_split(data,
                                                     labels,
                                                     train_size=0.8,
                                                     test_size=0.2,
                                                     random_state=seed,
                                                     stratify=labels)

    # Z-score normalization with train set's statistics.
    norm_train_data = (train_data-np.mean(train_data))/np.std(train_data)
    norm_test_data = (test_data-np.mean(train_data))/np.std(train_data)

    # PROCESS DATA & LABELS TO TRAIN AND TEST THE MODEL
    # Data need reshaping as the model gets
    # input = [samples, timesteps, features]
    train_data_seq = norm_train_data.reshape((len(norm_train_data),
                                              timesteps,
                                              len(norm_train_data[0])//timesteps))
    train_labels_seq = train_labels.reshape((len(norm_train_data), features))
    one_hot_lab_train = to_categorical(train_labels,num_classes)

    test_data_seq = norm_test_data.reshape((len(norm_test_data),
                                            timesteps,
                                            len(norm_test_data[0])//timesteps))
    test_labels_seq = test_labels.reshape((len(norm_test_data), features))
    one_hot_lab_test = to_categorical(test_labels, num_classes)

    # Define Constructor
    model = Sequential()

    # BUILD THE MODEL
    model.add(SimpleRNN(units=100,
                        activation='relu',
                        input_shape=(timesteps,
                                     len(norm_train_data[0])//timesteps),
                        dropout=0.0,
                        recurrent_dropout=0.0,
                        return_sequences=False))

    if (problem_type == 'binary') or (problem_type == 'multilabel'):
        model.add(Dense(num_classes, kernel_initializer='normal',
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.0),
                        activity_regularizer=regularizers.l2(0.0)))

    elif problem_type == 'multiclass':
        model.add(Dense(num_classes,
                        kernel_initializer='normal',
                        activation='softmax'))

    # COMPILE THE MODEL
    if problem_type == 'binary':
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    elif problem_type == 'multilabel':
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    elif problem_type == 'multiclass':
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

    # TRAIN THE MODEL
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_labels),
                                                      train_labels)
    tic = time.process_time()
    history = model.fit(train_data_seq,
                        one_hot_lab_train,
                        batch_size=25,
                        epochs=epochs,
                        validation_split=0.2,
                        class_weight=class_weights)
    toc = time.process_time()
    print(f'Elapsed time{np.round(toc-tic, 2)} seconds.')

    # Evaluate the model.
    test_predictions = model.predict_classes(test_data_seq)
    test_accuracy = accuracy_score(test_labels_seq, test_predictions)

    train_predictions = model.predict_classes(train_data_seq)
    train_accuracy = accuracy_score(train_labels_seq, train_predictions)

    # Confusion matrices
    train_conf_matr = confusion_matrix(train_labels, train_predictions)
    train_conf_matr = train_conf_matr/train_conf_matr.astype(np.float).sum(axis=1)

    test_conf_matr = confusion_matrix(test_labels, test_predictions)
    test_conf_matr = test_conf_matr / test_conf_matr.astype(np.float).sum(axis=1)

    output = {}
    output['model'] = model
    output['train_acc'] = train_accuracy
    output['test_acc'] = test_accuracy
    output['train_conf'] = train_conf_matr
    output['test_conf'] = test_conf_matr

    return output


def run_cnn_model(data, labels, epochs=20, num_classes=2,
                  problem_type='multiclass', seed=0):
    
    input_shape = (len(data[0]), 1)  # number of columns
    # Split the dataset into train and test sets.
    train_data, test_data, \
        train_labels, test_labels = train_test_split(data,
                                                     labels,
                                                     train_size=0.8,
                                                     test_size=0.2,
                                                     random_state=seed,
                                                     stratify=labels)

    # Z-score normalization with train set's statistics.
    norm_train_data = (train_data-np.mean(train_data))/np.std(train_data)
    norm_test_data = (test_data-np.mean(train_data))/np.std(train_data)

    # Build and compile the model.
    model = build_model(problem_type, num_classes, input_shape)

    # Data need reshaping as the model gets
    # input = [samples, timesteps, features]
    train_data_seq = np.expand_dims(norm_train_data, 2)
    train_labels_seq = train_labels.reshape((len(norm_train_data), 1))
    one_hot_lab_train = to_categorical(train_labels, num_classes)

    test_data_seq = np.expand_dims(norm_test_data, 2)
    test_labels_seq = test_labels.reshape((len(norm_test_data), 1))
    one_hot_lab_test = to_categorical(test_labels, num_classes)

    # Train the model
    tic = time.process_time()
    history = model.fit(train_data_seq,
                        one_hot_lab_train,
                        batch_size=50,
                        epochs=epochs,
                        validation_split=0.1)
    toc = time.process_time()
    print(f'Elapsed time{np.round(toc-tic, 2)} seconds.')

    # Evaluate the model.
    test_predictions = model.predict_classes(test_data_seq)
    test_accuracy = accuracy_score(test_labels_seq, test_predictions)

    train_predictions = model.predict_classes(train_data_seq)
    train_accuracy = accuracy_score(train_labels_seq, train_predictions)

    # Confusion matrices
    train_conf_matr = confusion_matrix(train_labels, train_predictions)
    train_conf_matr = train_conf_matr/train_conf_matr.astype(np.float).sum(axis=1)

    test_conf_matr = confusion_matrix(test_labels, test_predictions)
    test_conf_matr = test_conf_matr / test_conf_matr.astype(np.float).sum(axis=1)

    output = {}
    output['model'] = model
    output['train_acc'] = train_accuracy
    output['test_acc'] = test_accuracy
    output['train_conf'] = train_conf_matr
    output['test_conf'] = test_conf_matr

    return output


def run_lstm_model(data, labels, epochs=20, num_classes=2,
                   timesteps=2, features=1,
                   problem_type='multiclass', seed=0):
    # Split the dataset into train and test sets.
    train_data, test_data, \
        train_labels, test_labels = train_test_split(data,
                                                     labels,
                                                     train_size=0.8,
                                                     test_size=0.2,
                                                     random_state=seed,
                                                     stratify=labels)

    # Z-score normalization with train set's statistics.
    norm_train_data = (train_data-np.mean(train_data))/np.std(train_data)
    norm_test_data = (test_data-np.mean(train_data))/np.std(train_data)

    # Data need reshaping as the model gets input = [samples, timesteps, features]
    train_data_seq = norm_train_data.reshape((len(norm_train_data),
                                              timesteps,
                                              len(norm_train_data[0])//timesteps))
    train_labels_seq = train_labels.reshape((len(norm_train_data), 1))
    one_hot_lab_train = to_categorical(train_labels,num_classes)

    test_data_seq = norm_test_data.reshape((len(norm_test_data),
                                            timesteps,
                                            len(norm_test_data[0])//timesteps))
    test_labels_seq = test_labels.reshape((len(norm_test_data), 1))
    one_hot_lab_test = to_categorical(test_labels, num_classes)

    # DEFINE CONSTRUCTOR
    model = Sequential()

    # BUILD THE MODEL
    model.add(LSTM(units=100,
                   input_shape=(timesteps, len(norm_train_data[0])//timesteps),
                   activation='relu',
                   return_sequences=True))

    model.add(LSTM(units=100, activation='relu', dropout=0.0,
                   recurrent_dropout=0.0, return_sequences=False))
    if (problem_type == 'binary') or (problem_type == 'multilabel'):
        model.add(Dense(num_classes,
                        kernel_initializer='normal',
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.0),
                        activity_regularizer=regularizers.l2(0.0)))

    elif problem_type == 'multiclass':
        model.add(Dense(num_classes,
                        kernel_initializer='normal',
                        activation='softmax'))

    # COMPILE THE MODEL
    if problem_type == 'binary':
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    elif problem_type == 'multilabel':
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    elif problem_type == 'multiclass':
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    # TRAIN THE MODEL
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_labels),
                                                      train_labels)
    tic = time.process_time()
    history = model.fit(train_data_seq,
                        one_hot_lab_train,
                        epochs=epochs,
                        batch_size=25,
                        verbose=1,
                        validation_split=0.2,
                        class_weight=class_weights)
    toc = time.process_time()
    print(f'Elapsed time{np.round(toc-tic, 2)} seconds.')

    # Evaluate the model.
    test_predictions = model.predict_classes(test_data_seq)
    test_accuracy = accuracy_score(test_labels_seq, test_predictions)

    train_predictions = model.predict_classes(train_data_seq)
    train_accuracy = accuracy_score(train_labels_seq, train_predictions)

    # Confusion matrices
    train_conf_matr = confusion_matrix(train_labels, train_predictions)
    train_conf_matr = train_conf_matr/train_conf_matr.astype(np.float).sum(axis=1)

    test_conf_matr = confusion_matrix(test_labels, test_predictions)
    test_conf_matr = test_conf_matr / test_conf_matr.astype(np.float).sum(axis=1)

    output = {}
    output['model'] = model
    output['train_acc'] = train_accuracy
    output['test_acc'] = test_accuracy
    output['train_conf'] = train_conf_matr
    output['test_conf'] = test_conf_matr

    return output
