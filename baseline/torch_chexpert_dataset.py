from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ChexpertDataset(Dataset): 
  def __init__(self, r, hdf5_dataset, noisy_labels, batch_size, ground_truth, transform):    
    self.r = r # noise ratio
    self.hdf5_dataset = hdf5_dataset
    self.transform = transform
    self.noisy_labels = noisy_labels
    self.batch_size = batch_size
    self.ground_truth = ground_truth       
                
  def __getitem__(self, index):
    x = self.hdf5_dataset[index]
    img = Image.fromarray(x.astype(np.uint8))
    img = self.transform(img)
    target = self.noisy_labels[index]
    return img, target, self.ground_truth[index], index
           
  def __len__(self):
    return len(self.hdf5_dataset)


class ChexpertNLNLDataset(Datset):
  def __init__(self, r, np_dataset, noisy_labels, batch_size, transform):
    self.r = r # noise ratio
    self.np_dataset = np_dataset
    self.transform = transform
    self.noisy_labels = noisy_labels
    self.batch_size = batch_size
                
  def __getitem__(self, index):
    x = self.np_dataset[index]
    img = Image.fromarray(x.astype(np.uint8))
    img = self.transform(img)
    target = self.noisy_labels[index]
    return img, target, index
           
  def __len__(self):
    return len(self.np_dataset)