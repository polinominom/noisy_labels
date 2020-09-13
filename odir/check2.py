import h5py
import numpy as np
import os
import datetime
import pickle
import threading
from PIL import Image

with h5py.File('sem_dset.hdf5', 'a') as f:
    print(list(f.keys()))
    xx = 'train'
    print('on train')
    for t in range(5):
        y = t + 103
        a = f[xx][y]
        print(f[f'{xx}_label'][y])
        img = Image.fromarray(a.astype(np.uint8))
        img.save(f'./{xx}_{y}.png')
    print('on val')
    xx = 'val'
    for t in range(5):
        y = t+3
        a = f[xx][y]
        print(f[f'{xx}_label'][y])
        img = Image.fromarray(a.astype(np.uint8))
        img.save(f'./{xx}_{y}.png')