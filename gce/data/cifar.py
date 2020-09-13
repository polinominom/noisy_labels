from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from .utils import download_url, check_integrity, noisify
import logging
logging.basicConfig(filename='debug.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def unpickle(path):
    try:
        if sys.version_info[0] == 2:
            with open(path, 'rb') as fo:
                return pickle.load(fo)
        else:
            with open(path, 'rb') as fo:
                return pickle.load(fo, encoding='latin1')
    except Exception as ex:
        print('unable open path: %s due to the error: %s'%(path, str(ex)))
    

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, noise_folder='', random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.noise_type=noise_type
        self.noise_folder = noise_folder + 'cifar10/'
        self.nb_classes=10
        # now load the picked numpy arrays
        if self.train:
            noise_full_path = self.noise_folder
            if self.noise_type == 'symmetric':
                noise_full_path = noise_full_path + 'sym_exc_n%s_train_labels'%(str(int(noise_rate*100)))
            elif self.noise_type == 'pairflip':
                noise_full_path = noise_full_path + 'class_dependent_asym_n%s_train_labels'%(str(int(noise_rate*100)))
            
            self.train_data = unpickle(self.noise_folder + 'train_images')
            self.train_noisy_labels = unpickle(noise_full_path)
            #self.train_noisy_labels = self.train_noisy_labels.reshape(self.train_noisy_labels.shape[0], 1)

            self.train_labels =  unpickle(self.noise_folder + 'n00_train_labels')
            #self.train_labels = self.train_labels.reshape(self.train_labels.shape[0], 1)

            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            print('self.train_labels.shape: %s, self.train_noisy_labels: %s'%(self.train_labels.shape, self.train_noisy_labels.shape))
            print('loaded: %s'%(noise_full_path))
            print('actual noise: %s'%(str(np.sum(self.train_labels != self.train_noisy_labels)/self.train_noisy_labels.shape[0])))

        else:
            self.test_data = unpickle(self.noise_folder+'test_images')
            self.test_labels = unpickle(self.noise_folder+'test_labels')
            #self.test_labels = self.test_labels.reshape(self.test_labels.shape[0], 1)
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # msg = 'index: %s, self.train: %s, img.shape: %s, target.shape: %s '%(str(index), str(self.train), str(img.shape), str(target.shape))
        # print(msg)
        #logging.info(msg)
        
        img = Image.fromarray((img * 255).astype(np.uint8))
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CIFAR100(data.Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, noise_folder='',random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar100'
        self.noise_type=noise_type
        self.nb_classes=100
        self.noise_folder = noise_folder + 'cifar100/'

        # now load the picked numpy arrays
        if self.train:
            noise_full_path = self.noise_folder            
            if self.noise_type == 'symmetric':
                noise_full_path = noise_full_path + 'sym_exc_n%s_train_labels'%(str(int(noise_rate*100)))
            elif self.noise_type == 'pairflip':
                noise_full_path = noise_full_path + 'class_dependent_asym_n%s_train_labels'%(str(int(noise_rate*100)))
            
            self.train_data = unpickle(self.noise_folder + 'train_images')
            self.train_noisy_labels = unpickle(noise_full_path)
            self.train_labels =  unpickle(self.noise_folder + 'n00_train_labels')
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            print('self.train_labels.shape: %s, self.train_noisy_labels: %s'%(self.train_labels.shape, self.train_noisy_labels.shape))
            print('loaded: %s'%(noise_full_path))
            print('actual noise: %s'%(str(np.sum(self.train_labels != self.train_noisy_labels)/self.train_noisy_labels.shape[0])))
        else:
            self.test_data = unpickle(self.noise_folder+'test_images')
            self.test_labels = unpickle(self.noise_folder+'test_labels')
            
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type is not None:
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




