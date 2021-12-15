#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:57:33 2021

@author: spiros
"""

import pickle

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

# Add test data
# test_predictions = model.predict_classes(test_data_seq)
