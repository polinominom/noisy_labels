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


class ChexpertNLNLDataset(Dataset):
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


class MetricKeeper:
  def __init__(self, iteration_per_epoch, max_epoch, metric_name_list):
    self._dict = {}
    self.iteration_per_epoch = iteration_per_epoch
    self.max_epoch = max_epoch
    self.metric_name_list = metric_name_list
    for name in metric_name_list:
      self.reset(name)

  def reset(self, metric):
    self._dict[metric] = np.zeros(self.iteration_per_epoch*self.max_epoch)

  def add(self, metric, value, total_iteration):
    if not metric in self._dict.keys():
      print(f'unable to add a metric name: {metric} it does not exist in the class. Set it before use it.')
      return

    self._dict[metric][total_iteration] = value

  def get(self, metric, epoch):
    try:
      sub_dict = self._dict[metric]
      _from = self.iteration_per_epoch * epoch
      _to   = self.iteration_per_epoch * (epoch+1)
      summed_values = np.sum(sub_dict[_from:_to])
      return np.divide(summed_values, self.iteration_per_epoch)
    except Exception as e:
      print(f'unable to get a metric namely "{metric}" for the epoch "{epoch}" due to an error: {e}')

  def list_names(self):
    print(f'METRIC KEEPER - METRIC NAMES: {self.metric_name_list}')
    
  def reset_case(self, metric):
    n = self.iteration_per_epoch*self.max_epoch
    self._dict[metric] = np.zeros((n, 2))

  def add_case(self, metric, value, total_iteration):
    if not metric in self._dict.keys():
      print(f'unable to add a metric name: {metric} it does not exist in the class. Set it before use it.')
      return

    self._dict[metric][total_iteration] = [value, 1]

  def get_case(self, metric, epoch):
    try:
      sub_dict = self._dict[metric]
      _from = self.iteration_per_epoch * epoch
      _to   = self.iteration_per_epoch * (epoch+1)
      summed_values = np.sum(sub_dict[_from:_to][:,0])
      summed_counts = np.sum(sub_dict[_from:_to][:,1])
      return np.divide(summed_values, summed_counts)
    except Exception as e:
      print(f'unable to get a metric namely "{metric}" for the epoch "{epoch}" due to an error: {e}')

        