import h5py
import numpy as np
import pandas as pd
import os
import datetime
import pickle
import sys
from PIL import Image

# SAVE and LOAD files
def unpickle(fname):
  with open(fname, 'rb') as fp:
    return pickle.load(fp)

def print_remaining_time(before, currentPosition, totalSize):
  after = datetime.datetime.now()
  elaspsed_time = (after - before).seconds
  estimated_remaining_time = elaspsed_time * (totalSize - currentPosition) / currentPosition
  
  msg = f'{currentPosition}/{totalSize}({(100*currentPosition/totalSize):.2f}%) finished. Estimated Remaining Time: {estimated_remaining_time:.2f} seconds.'
  print(msg)

def create_file(length, fname = 'dset.hdf5'):
    open(fname, 'w+').close()
    os.remove(fname)
    f = h5py.File(fname, 'a')
    f.create_dataset("samples", (length, 224, 224, 3), compression="gzip", dtype='uint8')
    f.create_dataset("labels", (length,8),  dtype='uint8')
    f.create_dataset("global_indices", (length, 1), dtype='uint32')
    return f

# -- -- --
def put_data(h5_file, label_dict, length):
    df_all = pd.read_csv('./ground_truth/odir.csv')
    selected_indices = np.concatenate(list(label_dict.values()))
    print(f'selected_indices length: {len(selected_indices)}')
    # get the non_selected indices
    total_i = 0
    before = datetime.datetime.now()
    for i, v in enumerate(list(df_all['ID'])):
        if i in selected_indices:
            continue
        # non - selected sample found
        im = Image.open(f'./odir_train_treated_224/{v}')
        h5_file['samples'][total_i] = np.asarray(im.convert('RGB'), dtype=np.uint8)
        h5_file['labels'][total_i] = get_label(df_all, i)
        h5_file['global_indices'][total_i] = i
        print(f'global id: {i} - v:{v} - label:{get_label(df_all, i)}')
        print_remaining_time(before, total_i+1, length)
        total_i += 1
# -- --
def get_label(df, i):
    columns = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    ground_truth = np.zeros(8, dtype=np.uint8)
    for j, c in enumerate(columns):
        ground_truth[j] = df[c][i]
    return ground_truth
# -- --
def get_label_by_key(key):
    columns = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    ground_truth = np.zeros(8, dtype=np.uint8)
    for i, c in enumerate(columns):
        if c == key:
            ground_truth[i] = 1
    return ground_truth

label_dict = {}
with open('./sem_indices.txt', 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        arr = line.strip().split('::')
        label_name = arr[0]
        sample_indices = list(map(int, arr[1].split(':')[:-1]))
        label_dict[label_name] = sample_indices

print('label_dict has been obtained...')
fname = './odir_rest_dset.hdf5'
if not os.path.exists(fname):
    h5_file = create_file(6648, fname=fname)
    put_data(h5_file, label_dict, 6648)
    h5_file.close()
else:
    pass
    #with h5py.File(fname, 'a') as h5_file:
     #   put_data(h5_file, 'sem_val', _f=0)