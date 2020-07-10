import os
import sys
import datetime
import numpy as np
from shutil import copyfile
sys.path.append('../baseline')
sys.path.append('baseline')

import tf_chexpert_utilities


def print_remaining_time(before, currentPosition, totalSize):
  after = datetime.datetime.now()
  elaspsed_time = (after - before).seconds
  estimated_remaining_time = elaspsed_time * (totalSize - currentPosition) / currentPosition
  
  msg = '%i/%i(%.2f%s) finished. Estimated Remaining Time: %.2f seconds.'%(currentPosition, totalSize, (100*currentPosition/totalSize), '%' ,estimated_remaining_time)
  print(msg)

def get_file_name(i, t):
    idx = i//2
    if t == 'train':
        max_length = 5
    else:
        max_length = 4
    
    return '0'*(max_length - len(str(idx))) + str(idx) + '.pkl'

def check_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)

def check_class_dirs(name):
    healthy_dir = '%s/%s'%(name, 'healthy')
    check_dir(healthy_dir)

    not_healthy_dir = '%s/%s'%(name, 'nothealthy')
    check_dir(not_healthy_dir)

    return healthy_dir, not_healthy_dir

real_fnames = {}
generatable_folder_names = {}

##############
type_list = ['train', 'val', 'test']
train_fnames, val_fnames, test_fnames = tf_chexpert_utilities.get_fnames('./buffer/baseline')
print(train_fnames)
print(val_fnames)
print(test_fnames)
real_fnames['train']    = train_fnames
real_fnames['val']      = val_fnames
real_fnames['test']     = test_fnames
##############
folder_name = './buffer/forwardt'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

for t in type_list:
    name = '%s/%s'%(folder_name, t)
    generatable_folder_names[t] = name
    check_dir(name)
    healthy_dir, not_healthy_dir = check_class_dirs(name)
    before = datetime.datetime.now()
    for i, real_fname in enumerate(real_fnames[t]):
        source = real_fname
        if i%2==0:
            # copy to healthy
            destination = '%s/%s'%(healthy_dir, get_file_name(i, t))
        else:
            # copy to nothealth 
            destination = '%s/%s'%(not_healthy_dir, get_file_name(i, t))

        if (i+1)%100 == 0:
            print('-------- %s --------'%t)
            print_remaining_time(before, i+1, len(real_fnames[t]))
        
        copyfile(source, destination)



