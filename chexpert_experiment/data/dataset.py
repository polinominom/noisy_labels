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
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._labels = []
        self._mode = mode

        columns = ['Path','Sex','Age','Frontal/Lateral','AP/PA','No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
        self._label_header = columns[5:]
        df = pd.read_csv(label_path).fillna(0)
        self._image_paths = np.array(df[columns[0]])
        self._num_image = len(self._image_paths)
        # todo get labels
        for j in range(5, len(columns)):
            nd = np.array(df[columns[j]])
            self._labels.append(nd)
        self._labels = np.array(self._labels, dtype=np.int8).transpose()
        mask = self._labels==-1
        if cfg.label_fill_type == 'zeros':
            # assign every -1 to 0
            self._labels = np.where(mask, 0, self._labels)
        else:
            raise Exception('The label filling method namely [{}] is not implemented yet...')

        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
