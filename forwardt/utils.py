
import os
import sys
sys.path.append('../baseline')
sys.path.append('baseline')

import numpy as np

# NECESSARY FILES FROM BASELINE FOLDER
import tf_chexpert_utilities, tf_chexpert_callbacks, tf_chexpert_loader
import densenet

def _get_shuffled_indices(sample_fnames, seed):
  np.random.seed(seed)
  shuffled_idx_lst = np.arange(len(sample_fnames))
  np.random.shuffle(shuffled_idx_lst)
  return shuffled_idx_lst

def get_chexpert_loaders(noise, batch_size=32):
  print('Preparing ChexpertLoaders with noise: %s'%str(noise))
  sample_train_fnames, sample_val_fnames, sample_test_fnames = tf_chexpert_utilities.get_fnames('./buffer')
  print('-'*30)
  print(' LEN: %i, Absolute paths of train data: %s'%(len(sample_train_fnames),str(sample_train_fnames)))
  print(' LEN: %i, Absolute paths of val data: %s'%(len(sample_val_fnames), str(sample_val_fnames)))
  print(' LEN: %i, Absolute paths of test data: %s'%(len(sample_test_fnames), str(sample_test_fnames)))

  train_shuffled_idx_lst  = _get_shuffled_indices(sample_train_fnames, 1234)
  val_shuffled_idx_lst    = _get_shuffled_indices(sample_val_fnames, 5678)
  test_shuffled_idx_lst   = _get_shuffled_indices(sample_test_fnames, 9012)
  print('-'*30)
  print('shuffled indices for train: %s'%str(train_shuffled_idx_lst))
  print('shuffled indices for val: %s'%str(val_shuffled_idx_lst))
  print('shuffled indices for test: %s'%str(test_shuffled_idx_lst))
  
  """## GETTING THE GROUND TRUTH AND NOISY LABELS"""
  train_noisy_label_dict  = {}
  val_noisy_label_dict    = {}
  train_ground_truth      = tf_chexpert_utilities.get_noisy_labels(0.0, 'train', len(sample_train_fnames))
  val_ground_truth        = tf_chexpert_utilities.get_noisy_labels(0.0, 'val', len(sample_val_fnames))
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  for n in noises:
    train_noisy_label_dict[n] = tf_chexpert_utilities.get_noisy_labels(n, 'train', len(sample_train_fnames))
    val_noisy_label_dict[n]   = tf_chexpert_utilities.get_noisy_labels(n, 'val', len(sample_val_fnames))

  # GET SHUFFLED NAMES/LABES FOR TRAIN
  # TODO: can be re-oredered to have 0.0 in dictionary instead of outside of it
  train_s_fnames      = sample_train_fnames[train_shuffled_idx_lst]
  val_s_fnames        = sample_val_fnames[val_shuffled_idx_lst]
  train_ground_truth  = train_ground_truth[train_shuffled_idx_lst]
  val_ground_truth    = val_ground_truth[val_shuffled_idx_lst]
  for n in noises:
      train_noisy_label_dict[n]   =  train_noisy_label_dict[n][train_shuffled_idx_lst]
      val_noisy_label_dict[n]     =  train_noisy_label_dict[n][val_shuffled_idx_lst]

  if noise is None or noise == 0 or noise == 0.0:
    train_loader    = tf_chexpert_loader.ChexpertLoader(train_s_fnames, train_ground_truth, batch_size)
    val_loader      = tf_chexpert_loader.ChexpertLoader(val_s_fnames, val_ground_truth, batch_size)
  else:
    train_loader    = tf_chexpert_loader.ChexpertLoader(train_s_fnames, train_noisy_label_dict[noise], batch_size)
    val_loader      = tf_chexpert_loader.ChexpertLoader(val_s_fnames, val_noisy_label_dict[noise], batch_size)

  return train_loader, val_loader