import os
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
import argparse

def print_remaining_time(before, currentPosition, totalSize):
  after = datetime.datetime.now()
  elaspsed_time = (after - before).seconds
  estimated_remaining_time = elaspsed_time * (totalSize - currentPosition) / currentPosition
  
  msg = '%i/%i(%.2f%s) finished. Estimated Remaining Time: %.2f seconds.'%(currentPosition, totalSize, (100*currentPosition/totalSize), '%' ,estimated_remaining_time)
  print(msg)

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')
  parser.add_argument('--resume', type=int, default=0, help='Continue training')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the training, validation and test...')
  parser.add_argument('--max_epoch', type=int, default=150, help='MAXIMUM EPOH')
  parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
  parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
  opt = parser.parse_args() 
  return opt

def make_sure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        print('Given folder: %s does not exist. Starting to create it.'%folder_name)
        os.mkdir(folder_name)
        return

    print('Given folder: %s EXISTS.'%folder_name)

def get_time_ms():
    return int(round(time.time() * 1000))

def print_remaining_time(before, currentPosition, totalSize):
  after = datetime.datetime.now()
  elaspsed_time = (after - before).seconds
  estimated_remaining_time = elaspsed_time * (totalSize - currentPosition) / currentPosition
  
  msg = '%i/%i(%.2f%s) finished. Estimated Remaining Time: %.2f seconds.'%(currentPosition, totalSize, (100*currentPosition/totalSize), '%' ,estimated_remaining_time)
  print(msg)
def get_noisy_labels(noise_ratio, train_val, total_length):
  noisyCount = int(total_length * noise_ratio)
  noisy_labels = np.array([np.zeros(2,)]*total_length)
  for j in range(0, noisyCount):
    y = np.array([0.0,1.0])
    if j%2==0:
      y = np.array([1.0,0.0])
    noisy_labels[j] = y
  for j in range(noisyCount, total_length):
    y = np.array([0.0,1.0])
    if j%2==1:
      y = np.array([1.0,0.0])
    noisy_labels[j] = y
  return noisy_labels

# get full paths of all the files that are exist in the folder whose relative path is given as a param
def get_full_fnames(relative_path):
  lst = os.listdir(relative_path)
  return list(map(lambda x: '%s/%s'%(relative_path,x), lst))

def get_flipped_by_element(element):
  return np.array([abs(element[0]-1),abs(element[1]-1)])

# get the flipped versions of tbe given label list
def get_flipped_labels_from_lst(lst):
  # all elements of the given list should be in binary label form [1.0 0.0] or [0.0, 1.0]
  np.array(list(map(get_flipped_by_element, lst)))

# SAVE and LOAD files
def unpickle(fname):
  with open(fname, 'rb') as fp:
    return pickle.load(fp)

def save_ndarray(fname, data):
  with open(fname, 'wb') as fp:
    pickle.dump(data, fp)

# GROUND TRUTH CHECK -- uses threading
# Tries to check all of the ground truths if they are divided equally...
def get_gt_correct_list(t, fname_lst, corrected_lst, log):
  for i, v in enumerate(fname_lst):
    try:
      gt = unpickle(v)
      k = int(v.split('/')[-1])
      corrected_lst[k] = abs(int(gt[1]) - k%2) - abs(int(gt[0]) - k%2)
      if (i+1)%50==0 and log:
        print('[%i] finished: %i/%i ...'%(t, (i+1), len(fname_lst)))
    except Exception as e:
      print('*** error: %s'%str(e))


def get_fnames(sample_folder):
  sample_train_folder = '%s/%s'%(sample_folder, 'train')
  sample_val_folder   = '%s/%s'%(sample_folder, 'val')
  sample_test_folder  = '%s/%s'%(sample_folder, 'test')
  # Get relative file names
  train_dir = os.listdir(sample_train_folder)
  val_dir = os.listdir(sample_val_folder)
  test_dir = os.listdir(sample_test_folder)
  # remove macOS related unwanted files if they exist
  try:
    train_dir.remove('.DS_Store')
  except: 
    pass
  try:
    val_dir.remove('.DS_Store')
  except:
    pass
  try:
    test_dir.remove('.DS_Store')
  except:
    pass
  # sort
  train_dir = sorted(train_dir)
  val_dir = sorted(val_dir)
  test_dir = sorted(test_dir)
  # get full paths
  train_f = np.array(list(map(lambda x: "%s/%s"%(sample_train_folder, x), train_dir)))
  val_f = np.array(list(map(lambda x: "%s/%s"%(sample_val_folder, x), val_dir)))
  test_f = np.array(list(map(lambda x: "%s/%s"%(sample_test_folder, x), test_dir)))
  return train_f, val_f, test_f
