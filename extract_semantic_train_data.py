import os
import h5py
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import sys
sys.path.append('./baseline')
from tf_chexpert_utilities import *

def find_chunk_id(gi):
    return ( (gi//1000) + 1 ) * 1000

def find_local_id(gi):
    return gi%1000

def extract_samples(df, save_folder):
    gi_list = np.array(df['global_index'])
    li_list = np.array(list(map(find_local_id, gi_list)))
    ci_list = np.array(list(map(find_chunk_id, gi_list)))
    total_length = len(gi_list)

    save_fname = f'{save_folder}/sem_train.h5'
    # remove it if exists
    open(save_fname, 'w+').close()
    os.remove(save_fname)
    # get dset and create datasets
    f = h5py.File(save_fname, 'a')
    f.create_dataset("samples", (total_length, 224, 224, 3), compression="gzip", dtype="uint8")
    f.create_dataset("labels", (total_length,), compression="gzip", dtype="uint8")
    # 
    f['labels'][...] = np.array(df['class'], dtype=np.uint8)
    #sem_samples = np.array([k]*total_length, dtype=np.uint8)

    # getting samples
    before = datetime.datetime.now()
    for i, li in enumerate(li_list):
        ci = ci_list[i]
        # resize
        local_x = unpickle("./adjusted_data/adjusted_train_images_%i"%ci)
        x = local_x[li]
        img = Image.fromarray(x.astype(np.uint8))
        adjusted_im = img.resize((224, 224))
        # get back unsigned 8 bit integer version
        f["samples"][i] = np.asarray(adjusted_im.convert("RGB"),dtype=np.uint8)
        if (i+1)%5 == 0:
            print_remaining_time(before, i+1, total_length)

    f.close()


save_folder = './semantic_train_data'
fname = './semantic_train_indices.csv'
df_selected = pd.read_csv(fname)
extract_samples(df_selected, save_folder)