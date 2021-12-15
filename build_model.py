# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:47:07 2020.

@author: troullinou
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers


def build_model(problem_type, num_classes, input_shape):
    """
    Build the 1DCNN model.

    Parameters
    ----------
    problem_type : str
        DESCRIPTION.
    num_classes : int
        DESCRIPTION.
    input_shape : list
        DESCRIPTION.

    Returns
    -------
    model : keras model.
        DESCRIPTION.

    """
    # DEFINE CONSTRUCTOR
    model = Sequential()

    # BUILD THE MODEL
    model.add(Conv1D(filters=32,
                     kernel_size=10,
                     activation='relu',
                     input_shape=input_shape))

    model.add(Conv1D(filters=64, kernel_size=10,
                     activation='relu',
                     padding='valid'))

    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=64,
                     kernel_size=10,
                     activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    if (problem_type == 'binary') or (problem_type == 'multilabel'):
        model.add(Dense(num_classes, activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.0),
                        activity_regularizer=regularizers.l2(0.0)))

    elif problem_type == 'multiclass':
        model.add(Dense(num_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(0.0),
                        activity_regularizer=regularizers.l2(0.0)))
    # COMPILE THE MODEL
    adam = optimizers.Adam(learning_rate=0.001,
                           beta_1=0.9,
                           beta_2=0.999,
                           amsgrad=False)

    if problem_type == 'binary':
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    elif problem_type == 'multilabel':
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    elif problem_type == 'multiclass':
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

    return model
