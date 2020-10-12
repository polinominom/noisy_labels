import numpy as np
import tensorflow as tf
import pickle

class ChexpertLoader(tf.keras.utils.Sequence):
    """
        Loads dataset batch by batch to ram then deal with it
    """
    def __init__(self, hdf5_dataset, noisy_labels, real_ground_truth, batch_size):
        self.hdf5_dataset = hdf5_dataset
        self.labels = noisy_labels
        self.real_ground_truth = real_ground_truth
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.hdf5_dataset) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        _from = idx * self.batch_size
        _to = (idx+1) * self.batch_size
        data_x = self.hdf5_dataset[_from:_to]
        data_y = self.labels[_from:_to]
        return data_x, data_y

    def get_batch_length(self, idx):
        return self.batch_size

    def get_total_item_count(self):
        return len(self.hdf5_dataset)

    def unpickle(self, fname):
        with open(fname, 'rb') as fp:
            return pickle.load(fp)

    def get_all_samples(self):
        return self.hdf5_dataset

    def get_all_labels(self):
        return self.labels

    def get_all_real_ground_truth(self):
        return self.real_ground_truth
