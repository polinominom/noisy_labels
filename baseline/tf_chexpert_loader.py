import numpy as np
import tensorflow as tf
import pickle

class ChexpertLoader(tf.keras.utils.Sequence):
    """
        Loads dataset batch by batch to ram then deal with it
    """
    def __init__(self, image_fnames, ground_truth, batch_size):
        self.image_fnames = image_fnames
        self.labels = ground_truth
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_fnames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_fnames[idx * self.batch_size : (idx+1) * self.batch_size]
        data_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        k = np.zeros(256*256*3).reshape(256,256,3)
        data_x = np.array([k] * len(batch_x))
        del k

        for i in range(len(batch_x)):
            data_x[i] = self.unpickle(batch_x[i])

        return data_x, data_y

    def get_batch_length(self, idx):
        batch_x = self.image_fnames[idx * self.batch_size : (idx+1) * self.batch_size]
        return len(batch_x)

    def get_total_item_count(self):
        return len(self.image_fnames)

    def unpickle(self, fname):
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
