#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:35:05 2021

@author: hkab
"""
# Checkpoint
RESULT_PATH = '/content/drive/MyDrive/Colab Notebooks/Research/checkpoint'
# CHECKPOINT_PATH = RESULT_PATH + '/POS_Pretrained_phoBERT.pt'
# CHECKPOINT_PATH = RESULT_PATH + '/quantized_POS_Pretrained_phoBERT.pt'
# CHECKPOINT_PATH = RESULT_PATH + '/NER_Pretrained_phoBERT.pt'
CHECKPOINT_PATH = RESULT_PATH + '/quantized_NER_Pretrained_phoBERT.pt'

# MEASURE_MODEL_PATH = RESULT_PATH + '/POS_Pretrained_phoBERT.pt'
# MEASURE_MODEL_PATH = RESULT_PATH + '/quantized_POS_Pretrained_phoBERT.pt'
MEASURE_MODEL_PATH = RESULT_PATH + '/NER_Pretrained_phoBERT.pt'
# MEASURE_MODEL_PATH = RESULT_PATH + '/quantized_NER_Pretrained_phoBERT.pt'

# Training procedure
T_TOTAL = None
GLOBAL_STEP = 1
WARMUP_STEPS = 400
FINAL_THRESHOLD = 0.15
INITIAL_THRESHOLD = 1.0
FINAL_WARMUP = 2.0
INITIAL_WARMUP = 1.0
FINAL_LAMBDA = 1.0

MASKED_SCORE_LEARNING_RATE = 1e-2
LEARNING_RATE = 3e-5

REGULARIZATION = None