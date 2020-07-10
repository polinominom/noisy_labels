import os
import sys
sys.path.append('baseline')

import time
import json
import pickle
import datetime
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
# some local files
from tf_chexpert_loader import ChexpertLoader
from tf_chexpert_callbacks import EarlyStoppingAtMinLoss, PredictionSaveCallback
from tf_chexpert_utilities import *
import densenet

def fpath(folder, noise):
    return '%s/n_%s'%(folder, str(noise))

"""## model compile func"""
def compile_model(model, binary=False):
  # Instantiate a logistic loss function that expects integer targets.
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  # Instantiate an accuracy metric.
  accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  if binary:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    accuracy = tf.keras.metrics.binary_accuracy
  
  # Instantiate an optimizer.
  optimizer = tf.keras.optimizers.Adam()
  # Instantiate some callbacks
  
  model.compile(optimizer=optimizer, loss=loss,metrics=[accuracy])
  return model

"""## GET SAMPLE FILENAMES"""
sample_train_fnames, sample_val_fnames, sample_test_fnames = get_fnames('./buffer/baseline')

"""## SHUFFLING INDICES"""
# SHUFFLE TRAIN INDICES
np.random.seed(1234)
train_shuffled_idx_lst = np.arange(len(sample_train_fnames))
np.random.shuffle(train_shuffled_idx_lst)
print("shuffled_train_indices: %s"%str(train_shuffled_idx_lst))
# SHUFFLE VAL INDICES
np.random.seed(5678)
val_shuffled_idx_lst = np.arange(len(sample_val_fnames))
np.random.shuffle(val_shuffled_idx_lst)
print("shuffled_val_indices: %s"%str(val_shuffled_idx_lst))
# SHUFFLE TEST INDICES
np.random.seed(9012)
test_shuffled_idx_lst = np.arange(len(sample_test_fnames))
np.random.shuffle(test_shuffled_idx_lst)
print("shuffled_test_indices: %s"%str(test_shuffled_idx_lst))
print('-'*30)

"""## GETTING THE GROUND TRUTH AND NOISY LABELS"""
train_noisy_label_dict  = {}
val_noisy_label_dict    = {}
train_ground_truth      = get_noisy_labels(0.0, 'train', len(sample_train_fnames))
val_ground_truth        = get_noisy_labels(0.0, 'val', len(sample_val_fnames))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
noises = [0.1, 0.2, 0.3, 0.4, 0.5, .60]
for n in noises:
  train_noisy_label_dict[n] = get_noisy_labels(n, 'train', len(sample_train_fnames))
  val_noisy_label_dict[n]   = get_noisy_labels(n, 'val', len(sample_val_fnames))


"""## SOME TRAINING PARAMATERS"""
# get custom datasets for network training
BATCH_SIZE = 32
EPOCHS = 150 

# GET SHUFFLED NAMES/LABES FOR TRAIN
train_s_fnames      = sample_train_fnames[train_shuffled_idx_lst]
val_s_fnames        = sample_val_fnames[val_shuffled_idx_lst]
train_ground_truth  = train_ground_truth[train_shuffled_idx_lst]
val_ground_truth    = val_ground_truth[val_shuffled_idx_lst]
for n in noises:
    train_noisy_label_dict[n]   =  train_noisy_label_dict[n][train_shuffled_idx_lst]
    val_noisy_label_dict[n]     =  train_noisy_label_dict[n][val_shuffled_idx_lst]

tensorboard_log_dir = './tensorboard_logs'
network_training_pred_folder = './network_training_predictions'
prediction_save_folder = './network_training_predictions/densenet121_baseline'
model_dir = './models'
model_save_dir = './models/densenet121_baseline'

make_sure_folder_exists(tensorboard_log_dir)
make_sure_folder_exists(network_training_pred_folder)
make_sure_folder_exists(prediction_save_folder)
make_sure_folder_exists(model_dir)
make_sure_folder_exists(model_save_dir)
make_sure_folder_exists('%s/%s'%(prediction_save_folder, '00'))
for n in noises:
    make_sure_folder_exists('%s/%i'%(prediction_save_folder, int(n*100)))

opt = get_args()

if opt.noise_ratio == 0.0:
    train_loader    = ChexpertLoader(train_s_fnames, train_ground_truth, BATCH_SIZE)
    val_loader      = ChexpertLoader(val_s_fnames, val_ground_truth, BATCH_SIZE)
else:
    train_loader    = ChexpertLoader(train_s_fnames, train_noisy_label_dict[opt.noise_ratio], BATCH_SIZE)
    val_loader      = ChexpertLoader(val_s_fnames, val_noisy_label_dict[opt.noise_ratio], BATCH_SIZE)

# These callback initializations shouldn't give any errors.
monitor = 'val_loss'
mc_callback = ModelCheckpoint(fpath(model_save_dir, int(100*opt.noise_ratio)), monitor=monitor, verbose=1, save_best_only=True)
pcs = PredictionSaveCallback(train_loader, val_loader, prediction_save_folder='%s/%i'%(prediction_save_folder, int(100*opt.noise_ratio)))  
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
callbacks = [pcs, tensorboard_callback, mc_callback]

"""## TRAINING"""
print('best model weights will be saved at: %s'%str(fpath(model_save_dir, int(100*opt.noise_ratio))))
print('******************** TRAINING STARTED n:%s ********************'%str(opt.noise_ratio))
# print('EVERYTHING WORKS')
exit()
model = get_densenet()
model = compile_model(model, binary=True)
history = model.fit(train_loader, validation_data=val_loader, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks)

current_time_ms = lambda: int(round(time.time() * 1000))
json.dump(history.history, './history/%s_densenet121_%s.json'%(str(opt.noise_ratio),str(current_time_ms)))
