from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle 
from PIL import Image
def unpickle(fname):
  with open(fname, 'rb') as fp:
    return pickle.load(fp)

class ChexpertDataset(Dataset): 
  def __init__(self, r, fnames, noisy_labels, batch_size, ground_truth, transform):    
    self.r = r # noise ratio
    self.fnames = fnames
    self.transform = transform
    self.noisy_labels = noisy_labels
    self.batch_size = batch_size
    self.ground_truth = ground_truth       
                
  def __getitem__(self, index):
    x = unpickle(self.fnames[index])
    img = Image.fromarray(x)
    img = self.transform(img)
    target = self.noisy_labels[index]
    return img, target, self.ground_truth[index], index
           
  def __len__(self):
    return len(self.fnames)