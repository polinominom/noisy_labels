import re
import sys
import os
import argparse
import logging
import json
import time
import h5py
import subprocess
from shutil import copyfile
import torch
import scipy
import datetime
import pandas as pd
import numpy as np
import sklearn.covariance
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from easydict import EasyDict as edict
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel

def print_remaining_time(before, currentPosition, totalSize, additional=''):
  after = datetime.datetime.now()
  elaspsed_time = (after - before).seconds
  estimated_remaining_time = elaspsed_time * (totalSize - currentPosition) / currentPosition
  
  ratio = (100*currentPosition/totalSize)
  msg = f'{additional}{currentPosition}/{totalSize}({ratio:.2f}%) finished. Estimated Remaining Time: {estimated_remaining_time:.2f} seconds.'
  print("\r", end='')# this is needed to remov
  print(msg, end='', flush=True)

def random_sample_mean(feature, total_label, num_classes):
    
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered = False)    
    new_feature, fraction_list = [], []
    frac = 0.7
    sample_mean_per_class = torch.Tensor(num_classes, feature.size(1)).fill_(0)
    total_selected_list = []
    for i in range(num_classes):
        index_list = total_label[:,i].eq(1)
        temp_feature = feature[index_list.nonzero()[:,0]]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        #
        shuffler_idx = torch.randperm(temp_feature.size(0))
        index = shuffler_idx[:int(temp_feature.size(0)*frac)]
        fraction_list.append(int(temp_feature.size(0)*frac))
        total_selected_list.append(index_list.nonzero()[index])
        #
        selected_feature = torch.index_select(temp_feature, 0, index)
        new_feature.append(selected_feature)
        sample_mean_per_class[i].copy_(torch.mean(selected_feature, 0))
    
    total_covariance = 0
    for i in range(num_classes):
        flag = 0
        X = 0
        for j in range(fraction_list[i]):
            temp_feature = new_feature[i][j]
            temp_feature = temp_feature - sample_mean_per_class[i]
            temp_feature = temp_feature.view(-1,1)
            if flag  == 0:
                X = temp_feature.transpose(0,1)
                flag = 1
            else:
                X = torch.cat((X,temp_feature.transpose(0,1)),0)
            # find inverse            
        group_lasso.fit(X.cpu().numpy())
        inv_sample_conv = group_lasso.covariance_
        inv_sample_conv = torch.from_numpy(inv_sample_conv).float()
        if i == 0:
            total_covariance = inv_sample_conv*fraction_list[i]
        else:
            total_covariance += inv_sample_conv*fraction_list[i]
        total_covariance = total_covariance/sum(fraction_list)
    new_precision = scipy.linalg.pinvh(total_covariance.cpu().numpy())
    new_precision = torch.from_numpy(new_precision).float()
    
    return sample_mean_per_class, new_precision, total_selected_list

def MCD_single(feature, sample_mean, inverse_covariance, batch_size):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = batch_size
    total, mahalanobis_score = 0, 0
    frac = 0.7
    for data_index in range(int(np.ceil(feature.size(0)/temp_batch))):
        temp_feature = feature[total : total + temp_batch]
        gaussian_score = 0
        batch_sample_mean = sample_mean
        zero_f = temp_feature - batch_sample_mean
        term_gau = -0.5*torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
        # concat data
        if total == 0:
            mahalanobis_score = term_gau.view(-1,1)
        else:
            mahalanobis_score = torch.cat((mahalanobis_score, term_gau.view(-1,1)), 0)
        total += temp_batch
        
    mahalanobis_score = mahalanobis_score.view(-1)
    feature = feature.view(feature.size(0), -1)
    _, selected_idx = torch.topk(mahalanobis_score, int(feature.size(0)*frac))
    selected_feature = torch.index_select(feature, 0, selected_idx)
    new_sample_mean = torch.mean(selected_feature, 0)
    
    # compute covariance matrix
    X = 0
    flag = 0
    for j in range(selected_feature.size(0)):
        temp_feature = selected_feature[j]
        temp_feature = temp_feature - new_sample_mean
        temp_feature = temp_feature.view(-1,1)
        if flag  == 0:
            X = temp_feature.transpose(0,1)
            flag = 1
        else:
            X = torch.cat((X, temp_feature.transpose(0,1)),0)
    # find inverse            
    group_lasso.fit(X.cpu().numpy())
    new_sample_cov = group_lasso.covariance_
    
    return new_sample_mean, new_sample_cov, selected_idx

def make_validation(feature, total_label, sample_mean, inverse_covariance, num_classes, batch_size):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = batch_size
    total, mahalanobis_score, prediction = 0, 0, 0
    frac = 0.5
    for data_index in range(int(np.floor(feature.size(0)/temp_batch))):
        temp_feature = feature[total : total + temp_batch]
        temp_label = total_label[total : total + temp_batch]
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = temp_feature - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f.cuda(), inverse_covariance.cuda()), zero_f.t().cuda()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        generative_out = torch.index_select(gaussian_score.cpu(), 1, torch.LongTensor(list(range(13)))).diag()
        # concat data
        if total == 0:
            mahalanobis_score = generative_out
        else:
            mahalanobis_score = torch.cat((mahalanobis_score, generative_out), 0)
        total += temp_batch

    _, selected_idx = torch.topk(mahalanobis_score, int(total*frac))    
    return selected_idx

from torch.autograd import Variable
def train_weights(G_soft_list, total_val_data, total_val_label, batch_size):
    # loss function
    nllloss = F.binary_cross_entropy_with_logits
    # parameyer
    num_ensemble = len(G_soft_list)
    train_weights = torch.Tensor(num_ensemble, 1).fill_(1).cuda()
    train_weights = nn.Parameter(train_weights)
    total, correct_D = 0, 0
    optimizer = optim.Adam([train_weights], lr=0.02)
    total_epoch = 20
    total_num_data = total_val_data[0].size(0)

    for data_index in range(int(np.floor(total_num_data/batch_size))):
        target = total_val_label[total : total + batch_size].cuda()
        target = Variable(target)
        soft_weight = F.softmax(train_weights, dim=0)
        total_out = 0

        for i in range(num_ensemble):
            out_features = total_val_data[i][total : total + batch_size].cuda()
            out_features = Variable(out_features, volatile=True)
            feature_dim = out_features.size(1)
            output = F.softmax(G_soft_list[i](out_features) , dim=1)
                
            output = Variable(output.data, volatile=True)
            if i == 0:
                total_out = soft_weight[i]*output
            else:
                total_out += soft_weight[i]*output
                
        total += batch_size
        pred = torch.sigmoid(total_out).ge(0.5).float()
        equal_flag = pred.eq(target.data.float()).cuda()
        correct_D += equal_flag.sum()
            
    for epoch in range(total_epoch):
        total = 0
        shuffler_idx = torch.randperm(total_num_data)

        for data_index in range(int(np.floor(total_num_data/batch_size))):
            index = shuffler_idx[total : total + batch_size]
            target = torch.index_select(total_val_label, 0, index).cuda()
            target = Variable(target)
            total += batch_size

            def closure():
                optimizer.zero_grad()
                soft_weight = F.softmax(train_weights, dim=0)

                total_out = 0
                for i in range(num_ensemble):
                    out_features = torch.index_select(total_val_data[i], 0, index).cuda()
                    out_features = Variable(out_features)
                    feature_dim = out_features.size(1)
                    output = F.softmax(G_soft_list[i](out_features) , dim=1)
                    
                    if i == 0:
                        total_out = soft_weight[i]*output
                    else:
                        total_out += soft_weight[i]*output
                        
                total_out = torch.log(total_out + 10**(-10))
                loss = nllloss(total_out.float(), target.float())
                loss.backward()
                return loss

            optimizer.step(closure)
        
    correct_D, total = 0, 0
    
    for data_index in range(int(np.floor(total_num_data/batch_size))):
        target = total_val_label[total : total + batch_size].cuda()
        target = Variable(target)
        soft_weight = F.softmax(train_weights, dim=0)
        total_out = 0

        for i in range(num_ensemble):
            out_features = total_val_data[i][total : total + batch_size].cuda()
            out_features = Variable(out_features, volatile=True)
            feature_dim = out_features.size(1)
            output = F.softmax(G_soft_list[i](out_features) , dim=1)
                
            output = Variable(output.data, volatile=True)
            if i == 0:
                total_out = soft_weight[i]*output
            else:
                total_out += soft_weight[i]*output
                
        total += batch_size
        pred = torch.sigmoid(total_out).ge(0.5).float()
        equal_flag = pred.eq(target.data.float()).cuda()
        correct_D += equal_flag.sum()
        
    soft_weight = F.softmax(train_weights, dim=0)
    return soft_weight


def test_softmax(model, total_data, total_label, batch_size):
    with torch.no_grad():
        model.eval()
        total, correct_D = 0, 0
        iteration_count = int(np.floor(total_data.size(0)/batch_size))
        before = datetime.datetime.now()

        num_classes = total_label.size(1)
        for data_index in range(iteration_count):
            data, target = total_data[total : total + batch_size], total_label[total : total + batch_size]
            #data, target = Variable(data, volatile=True), Variable(target)
            total_out, _ = model(data)
            total += batch_size
            
            z = torch.zeros(num_classes, data.size(0))
            for i in range(num_classes):
                z[i] = total_out[i].squeeze(1)
            
            pred = torch.sigmoid(z.t()).ge(0.5).float()
            equal_flag = pred.eq(target.data.float()).cuda()
            correct_D += equal_flag.sum()/num_classes
            print_remaining_time(before, data_index+1, iteration_count, additional='[test_softmax]')
        return 100. * correct_D / total

def test_ensemble(G_soft_list, soft_weight, total_val_data, total_val_label, batch_size):
    with torch.no_grad():
        total, correct_D = 0, 0
        total_num_data = total_val_data[0].size(0)
        num_output = len(G_soft_list)
        num_clases = total_val_label.size(1)
        iteration_count = int(np.floor(total_num_data/batch_size))
        before = datetime.datetime.now()
        for data_index in range(iteration_count):
            target = total_val_label[total : total + batch_size].cuda()
            total_out = 0

            for i in range(num_output):
                out_features = total_val_data[i][total : total + batch_size].cuda()
                feature_dim = out_features.size(1)
                output = F.softmax(G_soft_list[i](out_features), dim=1)
                if i == 0:
                    total_out = soft_weight[i]*output
                else:
                    total_out += soft_weight[i]*output
                    
            total += batch_size
            pred = torch.sigmoid(total_out).ge(0.5).float()
            equal_flag = pred.eq(target.data.float()).cuda()
            correct_D += equal_flag.sum() / num_clases
            print_remaining_time(before, data_index + 1, iteration_count, additional='[test_ensemble]')
        return 100. * correct_D / total

def extract_features(device, model, dataloader, batch_size, file_root, data_name):
    with torch.no_grad():
        model.eval()
        #temp_x = torch.rand(2,3,320,320)
        #temp_list = model.backbone.features(temp_x)[1]
        # memory for saving the features
        total = 0
        steps = len(dataloader)
        dataiter = iter(dataloader)
        total_final_feature = [0]*steps
        before = datetime.datetime.now()
        for step in range(steps):
            data, _ = next(dataiter)
            data = data.to(device).float()
            #
            out_features = model.module.backbone.features(data)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            #
            if total == 0:
                total_final_feature = out_features.cuda().clone()
            else:
                total_final_feature = torch.cat((total_final_feature, out_features.cuda().clone()), 0)
            #
            total += 1
            #
            print_remaining_time(before, step+1, steps, additional='[extract_features]')
            
        file_name_data = '%s/%s_feature_%s.npy' % (file_root, data_name, str(1))
        n = total_final_feature.cpu().numpy()
        print(f'n.shape: {n.shape}')
        np.save(file_name_data , n)
def log_result(args, softmax_result, rog_result):
    #auc?
    contents = os.listdir('./chexpert_experiment')
    result_file_list = []
    for c in content:
        if 'rog_result_' in c:
            result_file_list.append(int(c.split('_')[-1]))
    m = -1
    if result_file_list != []:
        m = max(result_file_list)

    with open(f'./chexpert_experiment/rog_result_{m+1}.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: \t \t \t {v}\n')
        for k, v in softmax_result.items():
            f.write(f'{k}: \t \t \t {v}\n')
        for k, v in rog_result.items():
            f.write(f'{k}: \t \t \t {v}\n')

parser = argparse.ArgumentParser(description='rog')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str, help="Path of the config file in yaml format")
parser.add_argument('saved_path', default=None, metavar='SAVE_PATH', type=str, help="Path of the saved models")
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--train_chunks', type=str, help="h5 file path for train dataset")
parser.add_argument('--dev_h5', type=str, help="h5 file path for val dataset")
parser.add_argument('--dev_val_h5', type=str, help="h5 file path for dev_val dataset")
parser.add_argument('--chunk_count', type=int, default=0, help="usable chunk count")
parser.add_argument('--saved_model_path', type=str, default='', help='model which trained under noise.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help="Number of workers for each data loader")
parser.add_argument('--mode',type=str,help='inference mode, should be one of the following: [extract, run]')
args = parser.parse_args()
#print(args)
#print('args parsed correctly...')
#exit()
batch_size = args.batch_size
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))

#layer_list = list(range(num_output))

torch.cuda.manual_seed(0)
torch.cuda.set_device(args.gpu)

with h5py.File(args.dev_h5,'r') as dev_h5_file:
    np_dev_h5_file    = np.array(dev_h5_file['val'], dtype=np.uint8)
    np_dev_u_ones     = np.array(dev_h5_file['val_u_ones'], dtype=np.int8)    
    np_dev_u_zeros    = np.array(dev_h5_file['val_u_zeros'], dtype=np.int8)
    np_dev_u_random   = np.array(dev_h5_file['val_u_random'], dtype=np.int8)

with h5py.File(args.dev_val_h5,'r') as dev_val_h5_file:
    np_dev_val_h5_file    = np.array(dev_val_h5_file['val'], dtype=np.uint8)
    np_dev_val_u_ones     = np.array(dev_val_h5_file['val_u_ones'], dtype=np.int8)    
    np_dev_val_u_zeros    = np.array(dev_val_h5_file['val_u_zeros'], dtype=np.int8)
    np_dev_val_u_random   = np.array(dev_val_h5_file['val_u_random'], dtype=np.int8)

train_labels = {}
with h5py.File(f'{args.train_chunks}/train_labels.h5','r') as fp:
    train_labels['train_u_ones']    = np.array(fp['train_u_ones'], dtype=np.int8)
    train_labels['train_u_zeros']   = np.array(fp['train_u_zeros'], dtype=np.int8)
    train_labels['train_u_random']  = np.array(fp['train_u_random'], dtype=np.int8)

np_train_samples = None
for i in range(args.chunk_count):
    with open(f'{args.train_chunks}/chexpert_dset_chunk_{i+1}.npy', 'rb') as f:
        if np_train_samples is None:
            np_train_samples = np.load(f)
        else:
            np_train_samples = np.concatenate((np_train_samples, np.load(f)))
#
device = torch.device(f'cuda:{args.gpu}')
# load best chexpert model from normal
print('loading network: '+ args.saved_model_path)
model = Classifier(cfg)
#model = DataParallel(model, device_ids=args.gpu).to(device)
model = DataParallel(model, device_ids=[args.gpu]).to(device)
ckpt = torch.load(args.saved_model_path, map_location=device)
model.module.load_state_dict(ckpt['state_dict'])
model.cuda()
#
if args.mode == 'extract':
    dataloader_train = DataLoader(
        ImageDataset([np_train_samples, train_labels], cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    dataloader_dev_val = DataLoader(
        ImageDataset([np_dev_val_h5_file, np_dev_val_u_zeros, np_dev_val_u_ones, np_dev_val_u_random], cfg, mode='val'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    dataloader_dev = DataLoader(
        ImageDataset([np_dev_h5_file, np_dev_u_zeros, np_dev_u_ones, np_dev_u_random], cfg, mode='val'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    extract_features(device, model, dataloader_train, args.batch_size, args.saved_path, "inference_train_val")
    extract_features(device, model, dataloader_dev, args.batch_size, args.saved_path, "inference_test_test")
    extract_features(device, model, dataloader_dev_val, args.batch_size, args.saved_path, "inference_test_val")
elif args.mode == 'run':
    num_classes = 13
    test_data_list = []
    train_data_list = []
    test_val_data_list = []
    #
    file_root = args.saved_path
    #
    file_name_data = '%s/inference_test_test_feature_%s.npy' % (file_root, str(1))
    test_data = torch.from_numpy(np.load(file_name_data)).float()
    test_data_list.append(test_data)
    #
    file_name_data = '%s/inference_test_val_feature_%s.npy' % (file_root, str(1))
    test_data_val = torch.from_numpy(np.load(file_name_data)).float()
    test_val_data_list.append(test_data_val)
    # 
    file_name_data = '%s/inference_train_val_feature_%s.npy' % (file_root, str(1))
    train_data = torch.from_numpy(np.load(file_name_data)).float()
    train_data_list.append(train_data)
    # train data shape should be: (N, 1024)
    print(f'Train data shape: {train_data.shape}')
    if cfg.label_fill_type == 'ones':
        inference_train_labels              = torch.from_numpy(train_labels['train_u_ones'][:len(np_train_samples)]).float()
        inference_test_data_val_labels      = torch.from_numpy(np_dev_val_u_ones).float()
        inference_test_data_test_labels     = torch.from_numpy(np_dev_u_ones).float()
    elif cfg.label_fill_type == 'zeros':
        inference_train_labels              = torch.from_numpy(train_labels['train_u_zeros'][:len(np_train_samples)]).float()
        inference_test_data_val_labels      = torch.from_numpy(np_dev_val_u_zeros).float()
        inference_test_data_test_labels     = torch.from_numpy(np_dev_u_zeros).float()
    elif cfg.label_fill_type == 'random':
        inference_train_labels              = torch.from_numpy(train_labels['train_u_random'][:len(np_train_samples)]).float()
        inference_test_data_val_labels      = torch.from_numpy(np_dev_val_u_random).float()
        inference_test_data_test_labels     = torch.from_numpy(np_dev_u_random).float()
    #
    print('Random Sample Mean')
    sample_mean, sample_precision, _ = random_sample_mean(train_data, inference_train_labels, num_classes)
    #
    print('MCD_single for each class')
    #
    new_sample_mean_list = []
    new_sample_precision_list = []
    new_sample_mean = torch.Tensor(num_classes, train_data.size(1)).fill_(0).cpu()
    new_covariance = 0
    for i in range(num_classes):
        index_list = inference_train_labels[:,i].eq(1)
        temp_feature = train_data[index_list.nonzero()[:,0]]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        #
        temp_mean, temp_cov, _= MCD_single(temp_feature.cpu(), sample_mean[i], sample_precision, args.batch_size)
        new_sample_mean[i].copy_(temp_mean)
        if i == 0:
            new_covariance = temp_feature.size(0)*temp_cov
        else:
            new_covariance += temp_feature.size(0)*temp_cov
    print('Computing new covariance ...')
    new_covariance = new_covariance / train_data.size(0)
    new_precision = scipy.linalg.pinvh(new_covariance)
    new_precision = torch.from_numpy(new_precision).float().cuda()
    new_sample_mean_list.append(new_sample_mean)
    new_sample_precision_list.append(new_precision)
    #
    print('G SOFT LIST ...')
    G_soft_list = []
    target_mean = new_sample_mean_list 
    target_precision = new_sample_precision_list
    for i in range(len(new_sample_mean_list)):
        dim_feature = new_sample_mean_list[i].size(1)
        sample_w = torch.mm(target_mean[i].cuda(), target_precision[i].cuda())
        sample_b = -0.5*torch.mm(torch.mm(target_mean[i].cuda(), target_precision[i].cuda()), target_mean[i].t().cuda()).diag() + torch.Tensor(num_classes).fill_(np.log(1./num_classes)).cuda()
        G_soft_layer = nn.Linear(int(dim_feature), num_classes).cuda()
        G_soft_layer.weight.data.copy_(sample_w)
        G_soft_layer.bias.data.copy_(sample_b)
        G_soft_list.append(G_soft_layer.cpu())

    print('using small val (100)...')
    sel_index = -1
    selected_list = make_validation(test_data_val, inference_test_data_val_labels, 
                                    target_mean[sel_index], target_precision[sel_index], num_classes, args.batch_size)
    
    new_val_data_list = []
    for i in range(len(new_sample_mean_list)):
        # tese should stay CPU()
        new_val_data = torch.index_select(test_val_data_list[i], 0, selected_list.cpu())
        new_val_label = torch.index_select(inference_test_data_val_labels, 0, selected_list.cpu())
        new_val_data_list.append(new_val_data)
    print('Finding softweights...')
    soft_weight = train_weights(G_soft_list, new_val_data_list, new_val_label, args.batch_size)
    #
    soft_acc = test_softmax(model, np_dev_h5_file, inference_test_data_test_labels, args.batch_size)
    print(f'Softmax accuracy: {soft_acc}')
    #
    RoG_acc = test_ensemble(G_soft_list, soft_weight, [test_data], inference_test_data_test_labels, args.batch_size)
    print(f'RoG accuracy: {RoG_acc}')
    #
    log_result(args, {'soft_acc':soft_acc}, {'rog_acc':RoG_acc})