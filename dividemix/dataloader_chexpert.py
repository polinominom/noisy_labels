from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import sys
sys.path.append('./forwardt')
sys.path.append('./baseline')
from utils import get_chexpert_loaders
from tf_chexpert_utilities import *

class chexpert_dataset(Dataset): 
    def __init__(self, r, transform, mode, pred=[], probability=[], log=''):
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        train_loader, val_loader = get_chexpert_loaders(r, batch_size=32)

        if self.mode=='test':
            self.test_data = val_loader.get_all_samples()
            self.test_label = val_loader.get_all_real_ground_truth()
        else:    
            train_label = train_loader.get_all_real_ground_truth()
            train_data  = train_loader.get_all_samples()
            noise_label = train_loader.get_all_labels()
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
                self.probability = [probability[i] for i in pred_idx]   
                    
                clean = (np.array(noise_label)==np.array(train_label))                                                       
                auc_meter = AUCMeter()
                auc_meter.reset()
                auc_meter.add(probability,clean)        
                auc,_,_ = auc_meter.value()               
                log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                log.flush()      

                self.train_data = train_data[pred_idx]
                self.noise_label = noise_label[pred_idx]
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))                
            elif self.mode == "unlabeled":
                pred_idx = (1-pred).nonzero()[0]                                               
                self.train_data = train_data[pred_idx]
                self.noise_label = noise_label[pred_idx]
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img.astype(np.uint8))
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img.astype(np.uint8))
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img.astype(np.uint8))
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img.astype(np.uint8))
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class chexpert_dataloader():  
    def __init__(self, r, batch_size, transform_size, num_workers):
        self.r = r
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        mean = [0.4365, 0.4365, 0.4365]
        # get transform and initialize trainset and trainloader
        self.transform_train = transforms.Compose([transforms.Resize(transform_size),
                                            transforms.RandomCrop(transform_size, padding=4), 
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((mean[0], mean[1], mean[2]),(1.0, 1.0, 1.0))])
                                            
        self.transform_test = transforms.Compose([transforms.Resize(transform_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((mean[0], mean[1], mean[2]), (1.0, 1.0, 1.0))]) 
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = chexpert_dataset(r=self.r, transform=self.transform_train, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        #chexpert_dataset(Dataset): def __init__(self, r, transform, mode, pred=[], probability=[], log=''):                             
        elif mode=='train':
            labeled_dataset = chexpert_dataset(r=self.r, transform=self.transform_train, mode="labeled", pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = chexpert_dataset(r=self.r, transform=self.transform_train, mode="unlabeled", pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = chexpert_dataset(r=self.r, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = chexpert_dataset(r=self.r, transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        