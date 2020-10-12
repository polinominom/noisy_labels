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


def create_file(train_length, val_length, test_length, fname = 'dset.hdf5'):
    open(fname, 'w+').close()
    os.remove(fname)
    #dt = h5py.special_dtype(vlen=str)
    f = h5py.File(fname, 'a')
    f.create_dataset("train", (train_length, 224, 224, 3), compression="gzip", dtype='uint8')
    f.create_dataset("train_label", (train_length,8),  dtype='uint8')
    f.create_dataset("val", (val_length, 224, 224, 3), compression="gzip", dtype='uint8')
    f.create_dataset("val_label", (val_length,8),  dtype='uint8')
    #f.create_dataset("test", (test_length, 224, 224, 3), compression="gzip", dtype='uint8')
    #f.create_dataset("test_label", (test_length,8),  dtype=dt)
    return f

def get_data(label_dict, length_per_case_train, length_per_case_val):
    df_all = pd.read_csv('./ground_truth/odir.csv')
    result_dict_train = {}
    result_dict_val = {}
    for key,values in label_dict.items():
        result_dict_train[key] = np.zeros((length_per_case_train, 224, 224, 3))
        result_dict_val[key] = np.zeros((length_per_case_val, 224, 224, 3))
        for i, v in enumerate(values):
            fname = df_all['ID'][v]
            #print(f'getting {fname}')
            im = Image.open(f'./odir_train_treated_224/{fname}')
            if i >= length_per_case_train:
                j = i - length_per_case_train
                result_dict_val[key][j] = np.asarray(im.convert('RGB'), dtype=np.uint8)
            else:
                result_dict_train[key][i] = np.asarray(im.convert('RGB'), dtype=np.uint8)

    return result_dict_train, result_dict_val

def get_label_by_key(key):
    columns = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    ground_truth = np.zeros(8, dtype=np.uint8)
    for i, c in enumerate(columns):
        if c == key:
            ground_truth[i] = 1
            break
    return ground_truth
def put_data(h5_file, dict_train, dict_val):
    a = np.array(list(dict_train.values())).shape
    b = np.array(list(dict_val.values())).shape
    total_length_train  = a[0]*a[1]
    total_length_val    = b[0]*b[1]

    print('FILLING TRAIN')
    total_i = 0
    before = datetime.datetime.now()
    for key,values in dict_train.items():
        label = get_label_by_key(key)
        for i, v in enumerate(values):
            h5_file['train'][total_i] = v
            h5_file['train_label'][total_i] = label
            total_i += 1
            if (total_i+1)%3 == 0:
                print_remaining_time(before, total_i+1, total_length_train)
    
    print('FILLING VAL')
    total_i = 0
    before = datetime.datetime.now()
    for key,values in dict_val.items():
        label = get_label_by_key(key)
        for i, v in enumerate(values):
            h5_file['val'][total_i] = v
            h5_file['val_label'][total_i] = label
            total_i += 1
            if (total_i+1)%3 == 0:
                print_remaining_time(before, total_i+1, total_length_val)

label_dict = {}
with open('./sem_indices.txt', 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        arr = line.strip().split('::')
        label_name = arr[0]
        sample_indices = list(map(int, arr[1].split(':')[:-1]))
        label_dict[label_name] = sample_indices

print('label_dict has been obtained...')
fname = './sem_dset.hdf5'
if not os.path.exists(fname):
    r_train, r_val = get_data(label_dict, 41, 3)
    h5_file = create_file(328, 24, 1, fname=fname)
    put_data(h5_file, r_train, r_val)
    h5_file.close()
else:
    pass
    #with h5py.File(fname, 'a') as h5_file:
     #   put_data(h5_file, 'sem_val', _f=0)