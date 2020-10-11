
import os
import sys
sys.path.append('../baseline')
sys.path.append('baseline')
# NECESSARY FILES FROM BASELINE FOLDER
#import tf_chexpert_utilities, tf_chexpert_callbacks, tf_chexpert_loader
from tf_chexpert_utilities import *
from torch_chexpert_dataset import *
import h5py

import time
import math
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch

term_width = 10
term_width = int(term_width)

last_time = time.time()
begin_time = last_time
TOTAL_BAR_LENGTH = 65.

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def _get_shuffled_indices(sample_fnames, seed):
  np.random.seed(seed)
  shuffled_idx_lst = np.arange(len(sample_fnames))
  np.random.shuffle(shuffled_idx_lst)
  return shuffled_idx_lst

def get_noisy_gt(noise, correct_gt):
  if noise == 0:
    return correct_gt

  zero_indicces = np.where(correct_gt==0)[0]
  one_indices   = np.where(correct_gt==1)[0]
  result = np.array(list(correct_gt))
  one_indices   = one_indices[:int(len(one_indices)*noise)]
  zero_indicces = zero_indicces[:int(len(zero_indicces)*noise)]
  result[one_indices] = 0
  result[zero_indicces] = 1
  return result

def get_covid_testset(batch, seed, transofrm_test):
    with h5py.File('./buffer/baseline/covid_all_dset.hdf5', 'r') as fp:
        test_dset = np.array(fp['test'])
        test_labels = np.array(fp['test_labels'])

        np.random.seed(seed)
        shuffled_idx_lst = np.arange(len(test_dset))
        np.random.shuffle(shuffled_idx_lst)
        print("shuffled_train_indices: %s"%str(shuffled_idx_lst))
        testset = ChexpertDataset(r=0, hdf5_dataset=test_dset[shuffled_idx_lst], noisy_labels=test_labels[shuffled_idx_lst], batch_size=batch, ground_truth=test_labels[shuffled_idx_lst], transform=transofrm_test)
        return testset

def get_covid_datasets(noise, batch_size, transform_train, transform_test):
  """## GET SAVED HDF5 DATASET FILE"""    
  # get h5py DATASET file
  dset = h5py.File('./buffer/baseline/covid_baseline_dset.hdf5', 'r')
  train_dataset = dset['train']
  train_label_dataset = dset['train_labels']
  val_dataset = dset['val']
  val_label_dataset = dset['val_labels']

  #t = np.zeros((len(train_label_dataset), 2))
  #v = np.zeros((len(val_label_dataset), 2))
  
  # convert to categorical
  #for i, val in  enumerate(np.array(train_label_dataset)):
      #t[i][val] = 1
  #for i, val in  enumerate(np.array(val_label_dataset)):
      #v[i][val] = 1
  
  #train_label_dataset = t
  #val_label_dataset = v
  """## SHUFFLING INDICES"""
  # SHUFFLE TRAIN INDICES 
  np.random.seed(1034)
  train_shuffled_idx_lst = np.arange(len(train_dataset))
  np.random.shuffle(train_shuffled_idx_lst)
  print("shuffled_train_indices: %s"%str(train_shuffled_idx_lst))
  # SHUFFLE VAL INDICES
  np.random.seed(5678)
  val_shuffled_idx_lst = np.arange(len(val_dataset))
  np.random.shuffle(val_shuffled_idx_lst)
  print("shuffled_val_indices: %s"%str(val_shuffled_idx_lst))
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
  train_noisy_label_dict = {}
  val_noisy_label_dict = {}
  noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, .60]
  for n in noises:
    train_noisy_label_dict[n] = get_noisy_gt(n, np.array(train_label_dataset, dtype=np.int64))[train_shuffled_idx_lst]
    val_noisy_label_dict[n] = get_noisy_gt(n, np.array(val_label_dataset, dtype=np.int64))[val_shuffled_idx_lst]
  
  train_ground_truth  = np.array(train_label_dataset, dtype=np.int64)[train_shuffled_idx_lst]
  val_ground_truth    = np.array(val_label_dataset, dtype=np.int64)[val_shuffled_idx_lst]
  #
  np_train_dataset = np.array(train_dataset, dtype=np.uint8)[train_shuffled_idx_lst]
  np_val_dataset = np.array(val_dataset, dtype=np.uint8)[val_shuffled_idx_lst]
  #
  trainset  = ChexpertDataset(r=noise, hdf5_dataset=np_train_dataset, noisy_labels=train_noisy_label_dict[noise], batch_size=batch_size, ground_truth=train_ground_truth, transform=transform_train)
  valset    = ChexpertDataset(r=noise, hdf5_dataset=np_val_dataset, noisy_labels=val_noisy_label_dict[noise], batch_size=batch_size, ground_truth=val_ground_truth, transform=transform_test)
  return trainset, valset