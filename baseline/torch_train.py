import os
import sys
import h5py
import time
import json
import pickle
import datetime
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
#torch imports
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# some metrics
from sklearn.metrics import recall_score, precision_score, f1_score
# some local files
sys.path.append('baseline')
from tf_chexpert_utilities import *
from torch_chexpert_dataset import ChexpertDataset
# handle ssl problem
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

def fpath(folder, noise):
    return '%s/n_%s'%(folder, str(noise))

"""## model compile func"""
def compile_model(model, binary=False):
  # Instantiate a logistic loss function that expects integer targets.
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  # Instantiate an accuracy metric.
  accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  if binary:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    accuracy = tf.keras.metrics.binary_accuracy
  
  # Instantiate an optimizer.
  optimizer = tf.keras.optimizers.Adam()
  # Instantiate some callbacks
  
  model.compile(optimizer=optimizer, loss=loss,metrics=[accuracy])
  return model

def initialize_logger(fname):
    logger = h5py.File(fname, 'a')
    _createGroup('TRAIN', logger)
    _createGroup('VAL', logger)
    _createGroup('TEST', logger)
    _createSubGropus(logger['TRAIN'])
    _createSubGropus(logger['VAL'])
    _createSubGropus(logger['TEST'])
    return logger

def _createGroup(x, y): 
    if not x in y: 
        y.create_group(x)

def _createSubGropus(grp):
    _createGroup('softmax_pred', grp)
    _createGroup('train_loss', grp)
    _createGroup('real_train_loss', grp)
    _createGroup('train_acc', grp)
    _createGroup('real_train_acc', grp)
    _createGroup('precision', grp)
    _createGroup('recall', grp)
    _createGroup('f1_score', grp)

def _assign(_logger, EPOCH, data):
    if EPOCH in _logger:
        _logger[EPOCH][...] = data
    else:
        _logger[EPOCH] = data

def log_metrics(logger_file, KEY, EPOCH, softmax_pred, loss, real_loss, acc, real_acc, precision, recall, _f1Score): 
    if type(EPOCH) == type("str"):
        EPOCH = str(EPOCH)
    if not KEY in logger_file:
        print(f"KEY: {KEY} doesn't exist in the logger. ABORTING LOGGING...")
        return

    _assign(logger_file['%s/softmax_pred'%KEY], EPOCH=EPOCH, DATA=softmax_pred)
    _assign(logger_file['%s/loss'%KEY], EPOCH=EPOCH, DATA=loss)
    _assign(logger_file['%s/real_loss'%KEY], EPOCH=EPOCH, DATA=real_loss)
    _assign(logger_file['%s/acc'%KEY], EPOCH=EPOCH, DATA=acc)
    _assign(logger_file['%s/real_acc'%KEY], EPOCH=EPOCH, DATA=real_acc)
    _assign(logger_file['%s/precision'%KEY], EPOCH=EPOCH, DATA=precision)
    _assign(logger_file['%s/recall'%KEY], EPOCH=EPOCH, DATA=recall)
    _assign(logger_file['%s/f1_score'%KEY], EPOCH=EPOCH, DATA=_f1Score)
    

"""## GET SAMPLE FILENAMES"""
sample_train_fnames, sample_val_fnames, sample_test_fnames = get_fnames('./buffer/baseline')

"""## SHUFFLING INDICES"""
# SHUFFLE TRAIN INDICES
np.random.seed(1234)
train_shuffled_idx_lst = np.arange(len(sample_train_fnames))
np.random.shuffle(train_shuffled_idx_lst)
print("shuffled_train_indices: %s"%str(train_shuffled_idx_lst))
# SHUFFLE VAL INDICES
np.random.seed(5678)
val_shuffled_idx_lst = np.arange(len(sample_val_fnames))
np.random.shuffle(val_shuffled_idx_lst)
print("shuffled_val_indices: %s"%str(val_shuffled_idx_lst))
# SHUFFLE TEST INDICES
np.random.seed(9012)
test_shuffled_idx_lst = np.arange(len(sample_test_fnames))
np.random.shuffle(test_shuffled_idx_lst)
print("shuffled_test_indices: %s"%str(test_shuffled_idx_lst))
print('-'*30)

"""## GETTING THE GROUND TRUTH AND NOISY LABELS"""
train_noisy_label_dict  = {}
val_noisy_label_dict    = {}
train_ground_truth      = get_noisy_labels(0.0, 'train', len(sample_train_fnames))
val_ground_truth        = get_noisy_labels(0.0, 'val', len(sample_val_fnames))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - 
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, .60]
for n in noises:
  train_noisy_label_dict[n] = get_noisy_labels(n, 'train', len(sample_train_fnames))
  val_noisy_label_dict[n]   = get_noisy_labels(n, 'val', len(sample_val_fnames))


"""## SOME TRAINING PARAMATERS"""
# get custom datasets for network training

# GET SHUFFLED NAMES/LABES FOR TRAIN
train_s_fnames      = sample_train_fnames[train_shuffled_idx_lst]
val_s_fnames        = sample_val_fnames[val_shuffled_idx_lst]
train_ground_truth  = train_ground_truth[train_shuffled_idx_lst]
val_ground_truth    = val_ground_truth[val_shuffled_idx_lst]
for n in noises:
    train_noisy_label_dict[n]   =  train_noisy_label_dict[n][train_shuffled_idx_lst]
    val_noisy_label_dict[n]     =  train_noisy_label_dict[n][val_shuffled_idx_lst]

tensorboard_log_dir = './tensorboard_logs'
network_training_pred_folder = './network_training_predictions'
prediction_save_folder = './network_training_predictions/torch_baseline'
model_dir = './models'
model_save_dir = './models/torch_baseline'

make_sure_folder_exists(tensorboard_log_dir)
make_sure_folder_exists(network_training_pred_folder)
make_sure_folder_exists(prediction_save_folder)
make_sure_folder_exists(model_dir)
make_sure_folder_exists(model_save_dir)
make_sure_folder_exists('%s/%s'%(prediction_save_folder, '0'))
for n in noises:
    make_sure_folder_exists('%s/%i'%(prediction_save_folder, int(n*100)))

# get args
# # # # if you are using a notebook comment the code below and uncomment the ' NOETBOOK MODE ' code block# # # #
opt = get_args()

BATCH_SIZE = opt.batch_size
MAX_EPOCHS = opt.max_epoch
NOISE_RATIO = opt.noise_ratio

LR = opt.lr
MOMENTUM = opt.momentum
WEIGHT_DECAY = opt.weight_decay
RESUME = opt.resume
skip_mean = False
# # # # NOTEBOOK MODE # # # #
# BATCH_SIZE = 16
# MAX_EPOCHS = 150
# NOISE_RATIO = 0
# LR = 0.02
# MOMENTUM = 0.9
# WEIGHT_DECAY = 1e-4
# RESUME = 0
# skip_mean = True
# open h5py file to save the epoch states
logger = initialize_logger('%s/%i/log.hdf5'%(prediction_save_folder, int(100*NOISE_RATIO)))
if not skip_mean:
  # Get mean to initialize transform
  trainset = ChexpertDataset(r=NOISE_RATIO, fnames=train_s_fnames, noisy_labels=train_noisy_label_dict[NOISE_RATIO], batch_size=BATCH_SIZE, ground_truth=train_ground_truth, transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
  mean = 0
  before = datetime.datetime.now()    
  for i, data in enumerate(trainloader, 0):
      imgs, labels, real_labels, index= data
      mean += torch.from_numpy(np.mean(np.asarray(imgs), axis=(2,3))).sum(0)
      if (i+1)%15 == 0:
          print_remaining_time(before, i+1, len(trainloader))
  mean = mean / len(trainset)
else:
  mean = [0.5, 0.5, 0.5]
# get transform and initialize trainset and trainloader
transform_train = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomCrop(256, padding=4), 
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.ToTensor(),
                                      transforms.Normalize((mean[0], mean[1], mean[2]),(1.0, 1.0, 1.0))])
                                      
transform_test = transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize((mean[0], mean[1], mean[2]), (1.0, 1.0, 1.0))])

trainset = ChexpertDataset(r=NOISE_RATIO, fnames=train_s_fnames, noisy_labels=train_noisy_label_dict[NOISE_RATIO], batch_size=BATCH_SIZE, ground_truth=train_ground_truth, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

valset = ChexpertDataset(r=NOISE_RATIO, fnames=val_s_fnames, noisy_labels=val_noisy_label_dict[NOISE_RATIO], batch_size=BATCH_SIZE, ground_truth=val_ground_truth, transform=transform_test)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'train loader length: {len(trainloader)}')
print(f'val loader length: {len(valloader)}')
# get densenet
net = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)

# change the last classifier to predict 2 class instead of 10
num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(in_features=num_ftrs, out_features=2, bias=True)
# get criterion
criterion = nn.CrossEntropyLoss()

net.cuda()
criterion.cuda()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

num_classes = 2
train_preds = torch.zeros(len(trainset), num_classes) - 1.
val_preds = torch.zeros(len(valset), num_classes) - 1.



if RESUME == 1:
  fn = os.path.join(model_save_dir, 'checkpoint.pth.tar')
  ckpt = torch.load(fn)
  epoch_resume = ckpt['epoch']
  best_val_acc = ckpt['best_val_acc']
  net.load_state_dict(ckpt['state_dict'])
  optimizer.load_state_dict(ckpt['optimizer'])
  print('loading network SUCCESSFUL')
else:
  epoch_resume = 0
  best_val_acc = 0
  print('loading network FAILURE')

# TRAINING
for epoch in range(epoch_resume, MAX_EPOCHS):
  before = datetime.datetime.now()
  train_loss = 0
  train_acc = 0
  train_real_loss = 0
  train_real_acc = 0
  recall = 0
  precision = 0
  _f1Score = 0
  for i, data in enumerate(trainloader, 0):
    net.zero_grad()
    imgs, labels, real_ground_truths, index = data 
        # imgs               : data
        # labels             : noisy labels
        # real ground truths : not-noisy ground truths
        # index              : sample index
    imgs = Variable(imgs.cuda().float())
    labels = Variable(labels.cuda().long()[:,1])
    real_ground_truths = Variable(real_ground_truths.cuda().long())
        # forward
    logits = net(imgs.float())
        # get the accuracy and predictions of the current batch
    _, pred = torch.max(logits.data, -1)
    acc = float((pred==labels.data).sum()) 
    train_acc += acc
    train_real_acc += float((pred==real_ground_truths.data[:,1]).sum()) 
        # get current loss of the batch
    current_loss = criterion(logits, labels)
    train_loss += imgs.size(0) * current_loss.data    
    train_real_loss += imgs.size(0) * criterion(logits, real_ground_truths[:,1]).data
        # backward
    current_loss.backward()
    optimizer.step()
        # put prediction history in a list to save it later
    softmax_pred        = F.softmax(logits, -1).cpu().data
    train_preds[index.cpu()] = softmax_pred
        # put the metrics that COMPARES THE REAL GROUND TRUTHS
    recall    += imgs.size(0) * recall_score(real_ground_truths.cpu(), softmax_pred.cpu() > 0.5,    average="samples")
    precision += imgs.size(0) * precision_score(real_ground_truths.cpu(), softmax_pred.cpu() > 0.5, average="samples")
    _f1Score  += imgs.size(0) * f1_score(real_ground_truths.cpu(), softmax_pred.cpu() > 0.5,        average="samples")
    if (i+1)%100==0:
      print_remaining_time(before, i+1, len(trainloader))
        
  # get avarage loss and accuracy
  train_loss  /= len(trainset)
  train_acc   /= len(trainset)
  recall      /= len(trainset)
  precision   /= len(trainset)
  _f1Score    /= len(trainset)
  train_real_loss /= len(trainset)
  train_real_acc  /= len(trainset)
  # HANLE LOGS
  log_metrics(
    logger, 
    KEY='TRAIN', 
    EPOCH=epoch, 
    softmax_pred=train_preds, 
    loss=train_loss.cpu(),
    real_loss=train_real_loss.cpu(),
    acc=train_acc,
    real_acc=train_real_acc,
    recall=recall,
    precision=precision,
    _f1Score = _f1Score)

  print('[%6d/%6d] loss: %5f, real_loss: %5f, acc: %5f, real_acc:%5f, recall: %5f, precision: %5f, f1_score: %5f' %(epoch, MAX_EPOCHS, train_loss, train_real_loss, train_acc, train_real_acc, recall, precision, _f1Score))
    # reset the placeholder for the predictions in the next epoch 
  train_preds = train_preds*0 - 1.
    # VALIDATION STATE
  net.eval()
  val_loss = 0.0
  val_real_loss = 0.0
  val_acc = 0.0
  val_real_acc = 0.0
  val_prec = 0
  val_f1_score = 0
  val_recall = 0
  before = datetime.datetime.now()
  with torch.no_grad():
    for i, data in enumerate(valloader, 0):
      imgs, labels, real_ground_truths, index = data
      imgs = Variable(imgs.cuda())
      labels = Variable(labels.cuda().long()[:,1])
      real_ground_truths = Variable(real_ground_truths.cuda().long())
            # LOSS and REAL LOSS
      logits = net(imgs.float())
      loss = criterion(logits, labels)
      val_loss += imgs.size(0)*loss.data
      val_real_loss += imgs.size(0) * criterion(logits, real_ground_truths[:,1]).data
            # ACC and REAL ACC
      _, pred = torch.max(logits.data, -1)
      acc = float((pred==labels.data).sum())
      val_acc += acc
      val_real_acc += float((pred==real_ground_truths.data[:,1]).sum()) 
            # put prediction history in a list to save it later
      softmax_pred            = F.softmax(logits, -1).cpu().data
      val_preds[index.cpu()]  = softmax_pred
            # put the metrics that COMPARES THE REAL GROUND TRUTHS
      val_recall    += imgs.size(0) * recall_score(real_ground_truths.cpu(), softmax_pred.cpu() > 0.5,    average="samples")
      val_prec      += imgs.size(0) * precision_score(real_ground_truths.cpu(), softmax_pred.cpu() > 0.5, average="samples")
      val_f1_score  += imgs.size(0) * f1_score(real_ground_truths.cpu(), softmax_pred.cpu() > 0.5,        average="samples")
      if (i+1)%15 == 0:
        print_remaining_time(before, i+1, len(trainloader))

  val_loss /= len(valset)
  val_acc  /= len(valset)
    ####################################################
    # log the validation
  log_metrics(
    logger, 
    KEY='VAL', 
    EPOCH=epoch, 
    softmax_pred=val_preds, 
    loss=val_loss.cpu(),
    real_loss=val_real_loss.cpu(),
    acc=val_acc,
    real_acc=val_real_acc,
    recall=val_recall,
    precision=val_prec,
    _f1Score=val_f1_score)
  ####################################################
  print('\tVALIDATION...loss: %5f, acc: %5f, best_acc: %5f, real_acc: %5f, real_loss: %5f, recall: %5f, precision: %5f f1_score: %5f'%(val_loss, val_acc, best_val_acc, val_real_acc, val_real_loss, val_recall, val_prec, val_f1_score))
    ####################################################
  net.train()
    ####################################################
  val_preds = val_preds * 0 - 1.

  best_val_acc = max(val_acc, best_val_acc)
  is_best = val_acc > best_val_acc

  state = ({'epoch' : epoch,'state_dict' : net.state_dict(),'optimizer' : optimizer.state_dict(),'best_val_acc' : best_val_acc})

  fn = os.path.join(model_save_dir, 'checkpoint.pth.tar')
  best_fn = os.path.join(model_save_dir, 'best_val_checkpoint.pth.tar')
  print('saving model...')
  torch.save(state, fn)
  # save state of the best validation    
  if is_best:
    print('saving best val acc model...')
    torch.save(state, best_fn)



logger.close()