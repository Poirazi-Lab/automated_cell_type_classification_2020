#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:57:33 2021

@author: spiros
"""

import pickle
import tensorflow as tf
from tensorflow import keras
from models import run_cnn_model

# ----------------------------------------------------------------------------
# Load a dataset
with open('dataset.pkl', 'rb') as handle:
    DATA = pickle.load(handle)

data = DATA['data']
labels = DATA['labels']
# ----------------------------------------------------------------------------

# Train the model
output = run_cnn_model(data, labels, epochs=20,
                       num_classes=4, problem_type='multiclass', seed=0)

# Add test data and make predicitions.
model = output['model']
test_data_seq = tf.expand_dims(test_data_seq, axis=-1)
test_predictions = model.predict_classes(test_data_seq)  # returns the predicted labels.

# In case that we have the test labels.
test_labels = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)  # if any
scores = model.evaluate(test_data_seq, test_labels, verbose=0)  # returns a list, [loss, accuracy]

