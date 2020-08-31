
import os
import sys
sys.path.append('../baseline')
sys.path.append('baseline')

import numpy as np

# NECESSARY FILES FROM BASELINE FOLDER
#import tf_chexpert_utilities, tf_chexpert_callbacks, tf_chexpert_loader
from tf_chexpert_utilities import *
from tf_chexpert_loader import *
import densenet
import h5py

def _get_shuffled_indices(sample_fnames, seed):
  np.random.seed(seed)
  shuffled_idx_lst = np.arange(len(sample_fnames))
  np.random.shuffle(shuffled_idx_lst)
  return shuffled_idx_lst

def get_chexpert_loaders(noise, batch_size=32):
  print('Preparing ChexpertLoaders with noise: %s'%str(noise))
  #sample_train_fnames, sample_val_fnames, sample_test_fnames = tf_chexpert_utilities.get_fnames('./buffer')
  dset = h5py.File('./buffer/baseline/dset.hdf5', 'r')
  train_dataset = dset['train']
  val_dataset = dset['val']
  test_dataset = dset['test']
  # shuffle
  train_shuffled_idx_lst  = _get_shuffled_indices(train_dataset, 1234)
  val_shuffled_idx_lst    = _get_shuffled_indices(val_dataset, 5678)
  test_shuffled_idx_lst   = _get_shuffled_indices(test_dataset, 9012)
  print('-'*30)
  print('shuffled indices for train: %s'%str(train_shuffled_idx_lst))
  print('shuffled indices for val: %s'%str(val_shuffled_idx_lst))
  print('shuffled indices for test: %s'%str(test_shuffled_idx_lst))
  
  """## GETTING THE GROUND TRUTH AND NOISY LABELS"""
  train_noisy_label_dict  = {}
  val_noisy_label_dict    = {}
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, .60]
  for n in noises:
    train_noisy_label_dict[n] = get_noisy_labels(n, 'train', len(train_dataset))
    val_noisy_label_dict[n]   = get_noisy_labels(n, 'val', len(val_dataset))

  print('getting train data in memory...')
  np_train_dataset = np.array(train_dataset, dtype=np.uint8)
  print('getting validation data in memory...')
  np_val_dataset = np.array(val_dataset, dtype=np.uint8)
  #print('getting test data in memory...')
  #np_test_dataset = np.array(test_dataset, dtype=np.uint8)

  # GET SHUFFLED NAMES/LABES FOR TRAIN
  np_train_dataset      = np_train_dataset[train_shuffled_idx_lst]
  np_val_dataset        = np_val_dataset[val_shuffled_idx_lst]
  #np_test_dataset  = np_test_dataset[test_shuffled_idx_lst]
  for n in noises:
      train_noisy_label_dict[n]   =  train_noisy_label_dict[n][train_shuffled_idx_lst]
      val_noisy_label_dict[n]     =  val_noisy_label_dict[n][val_shuffled_idx_lst]
  train_gt = train_noisy_label_dict[0]
  val_gt = val_noisy_label_dict[0]

  train_loader    = ChexpertLoader(np_train_dataset,  train_noisy_label_dict[noise], train_gt, batch_size)
  val_loader      = ChexpertLoader(np_val_dataset, val_noisy_label_dict[noise], val_gt, batch_size)
  return train_loader, val_loader