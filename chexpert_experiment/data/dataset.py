import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
import pandas as pd
np.random.seed(0)

class ImageDataset(Dataset):
    def __init__(self, x_lst, cfg, mode='train'):
        self.cfg = cfg
        self._mode = mode
        self._images = x_lst[0]
        self._num_image = len(self._images)
        self._chunk_length = 14000
        self._chunk_id = None
        
        if cfg.label_fill_type == 'zeros':
            self._labels = x_lst[1]
            if mode == 'train':
                self._labels = x_lst[1]['train_u_zeros']
        elif cfg.label_fill_type == 'ones':
            self._labels = x_lst[2]
            if mode == 'train':
                self._labels = x_lst[1]['train_u_ones']
        elif cfg.label_fill_type == 'random':
            self._labels = x_lst[3]
            if mode == 'train':
                self._labels = x_lst[1]['train_u_random']
        else:
            raise Exception(f'The label filling method namely [{cfg.label_fill_type}] is not implemented yet...')

       if mode == 'train':
            self._all_labels = np.array(self._labels)
            self._labels = self._labels[:self._num_image]

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        im = self._images[idx]
        image = Image.fromarray(im)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = None
        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
