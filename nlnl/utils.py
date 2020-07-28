
import os
import sys
sys.path.append('../baseline')
sys.path.append('baseline')

import numpy as np

# NECESSARY FILES FROM BASELINE FOLDER
#import tf_chexpert_utilities, tf_chexpert_callbacks, tf_chexpert_loader
from tf_chexpert_utilities import *
from tf_chexpert_dataset import *
import h5py

seed_dict= {'train':1234, 'val':5678, 'test':9012}
def _get_shuffled_indices(sample_length, seed):
  np.random.seed(seed)
  shuffled_idx_lst = np.arange(sample_length)
  np.random.shuffle(shuffled_idx_lst)
  return shuffled_idx_lst

def get_labels(h5_file, noise_ratio, mode, shuffled=True, categorical=True):
    print(f'nlnl.utils.get_labels called with params: noise_ratio:{noise_ratio}, mode:{mode}, shuffled:{shuffled}, categorical:{categorical}')
    if mode != 'train' or mode != 'val' or mode != 'test':
        print(f'Unexpected mode received. expected: "train, val, or test" received: {mode}')
        return

    dataset = h5_file[mode]
    lenght_dataset = len(dataset)
    labels = get_noisy_labels(noise_ratio, mode, lenght_dataset)
    if shuffled:
        shuffled_idx_lst = _get_shuffled_indices(lenght_dataset, seed_dict[mode])
        labels = labels[shuffled_idx_lst]
    if not categorical:
        labels = labels[:,1]
    return labels

def get_datasets(h5_file, noise_ratio, batch_size, mode, transform, shuffled=True):
    print(f'nlnl.utils.get_dataset called with params: noise_ratio:{noise_ratio}, mode:{mode}, shuffled:{shuffled}')
    if mode != 'train' or mode != 'val' or mode != 'test':
        print(f'Unexpected mode received. expected: "train, val, or test" received: {mode}')
        return
    dataset = h5_file[mode]
    print(f'LEN: {len(dataset)}, Absolute paths of {mode} data: {str(dataset)}')
    lenght_dataset = len(dataset)
    print(f'getting {mode} data in memory...')
    np_dataset = np.array(dataset, dtype=np.uint8)
    if shuffled:
        shuffled_idx_lst = _get_shuffled_indices(lenght_dataset, seed_dict[mode])
        np_dataset = np_dataset[shuffled_idx_lst]

    labels = get_labels(h5_file, noise_ratio, mode, shuffled=shuffled, categorical=False)
    nlnlDataset = ChexpertNLNLDataset(r=noise_ratio, np_dataset=np_dataset, noisy_labels=labels, 
                                        batch_size=batch_size, transform=transform)
    return nlnlDataset