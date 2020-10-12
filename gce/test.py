import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import csv
from models.densenet import *
from utils import *
from data.cifar import CIFAR10, CIFAR100
from TruncatedLoss import TruncatedLoss
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser(description='PyTorch TruncatedLoss')
parser.add_argument('--batch_size', '-b', default=10,type=int, help='mini-batch size (default: 10)')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--seed', type = int, help = 'random seed between 0 and 10000', default = 3256)
parser.add_argument('--k', type = float, help = 'hyperparameter for gce', default = 0.3)
parser.add_argument('--q', type = float, help = 'hyperparameter for gce', default = 0.5)
parser.add_argument('--checkpoint_folder', type=str, help='checekpoint folder', default='covid_gce_checkpoint')
best_acc = 0

args = parser.parse_args()

def main():
    use_cuda = torch.cuda.is_available()
    global best_acc 
    
    # from utils
    TRANSFORM_RESIZE = 224
    mean = [0.4253, 0.4249, 0.4248]
    std = [0.3705, 0.3704, 0.3703]

    transform_train = transforms.Compose([transforms.Resize(TRANSFORM_RESIZE),
        transforms.RandomCrop(TRANSFORM_RESIZE, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((mean[0], mean[1], mean[2]),(std[0],std[1],std[2]))])

    transform_test = transforms.Compose([transforms.Resize(TRANSFORM_RESIZE),
        transforms.ToTensor(),
        transforms.Normalize((mean[0], mean[1], mean[2]), (std[0],std[1],std[2]))])

    trainset, valset = get_covid_datasets(args.noise_rate, args.batch_size, transform_train, transform_test)
    testset = get_covid_testset(args.batch_size, args.seed, transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            imgs, labels, real_ground_truths, index = data
            positive_indices = np.where(labels.data.cpu() == 0)[0]
            negative_indices = np.where(labels.data.cpu() == 1)[0]
            if len(negative_indices) == 0:
                print('train WRONG negative_indices')
                print(i)
            if len(positive_indices) == 0:
                print('train WRONG positive_indices')
                print(i)

    # Load checkpoint.
    print('==> Loading checkpoint..')
    assert os.path.isdir(args.checkpoint_folder), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./{args.checkpoint_folder}/current_net_n_0_k_{int(args.k*100)}_q_{int(args.q*100)}.h5', map_location=lambda storage, loc: storage)
    net = checkpoint['current_net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
    
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')
    else:
        net = net.cpu().double()

    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()
    metrics = val(0, testloader, net, criterion)
    metric_name_lst = ['loss', 'acc', 'recall', 'precision', 'auc', 'f_one', 'positive acc', 'negative acc']
    for j, name in enumerate(metric_name_lst):
        print(f'{name}: {metrics[j]}')
        

def val(epoch, valloader, net, criterion):
    global best_acc
    net.eval()

    val_loss = 0
    correct = 0
    total = 0
    total_positive = 0
    total_negative = 0
    
    positive_acc = 0
    negative_acc = 0
    val_recall = 0
    val_auc = 0
    val_precision = 0
    val_f_one_score = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valloader):
            inputs, targets, real_ground_truths, indexes = data
            inputs = inputs.cuda()
            targets = targets.cuda()
            real_ground_truths = real_ground_truths.cuda()
            indexes = indexes.cuda()

            predictions = net(inputs)
            predictions = predictions.cpu()
            softmax_pred = F.softmax(predictions, dim=1).cpu().data
            _, pred = torch.max(softmax_pred, 1)
            labels = real_ground_truths.cpu().data
            total_correct = (pred == labels).sum()
            positive_indices = np.where(labels == 1)[0]
            negative_indices = np.where(labels == 0)[0]
            #print(positive_indices)
# ------ # ------ # ------ # ------ # ------ # ------ # ------ #
#                | Positive Prediction | Negative Prediction
# Positive Class | True Positive (TP)  | False Negative (FN)
# Negative Class | False Positive (FP) | True Negative (TN)
# ------ # ------ # ------ # ------ # ------ # ------ # ------ #
            true_positives  = (pred[positive_indices]==labels[positive_indices]).sum()
            true_negatives  = (pred[negative_indices]==labels[negative_indices]).sum()
            false_positives = (pred[positive_indices]==0).sum()
            false_negatives = (pred[negative_indices]==1).sum()
            #print('*'*30)
            #print(f'TP: {true_positives}')
            #print(f'TN: {true_negatives}')
            #print(f'FP: {false_positives}')
            #print(f'FN: {false_negatives}')
            #print('*'*30)
            #print(f'total positive: {len(positive_indices)}')
            #print(f'total negative: {len(negative_indices)}')
            s_confusion_m = f'[TP:{true_positives}][TN:{true_negatives}][FP:{false_positives}][FN:{false_negatives}][t_pos:{len(positive_indices)}[t_neg:{len(negative_indices)}]'
            # Precision = TruePositives / (TruePositives + FalsePositives)
            prec = 1. * true_positives/(true_positives+false_positives)
            s_prec = f'{prec:.3f}'
            #print(f'prec: {prec:.3f}')
            # recall = true positives / true positives + false negatives
            reca = 1. * true_positives / (true_positives+false_negatives)
            s_reca = f'{reca:.3f}'
            #print(f'reca: {reca:.3f}')
            # f-measure = (2 * Precision * Recall) / (Precision + Recall)
            Fscr = 1. * (2*reca*prec) / (reca+prec)
            s_fscr = f'{Fscr:.3f}'
            #print(f'Fscr: {Fscr:.3f}')
            # a = tp
            # b = fp
            # c = fn
            # d = TN
            # auc TP/2(TP+FN)+TN/2(FP+TN)
            T = 1. * true_positives/(2*true_positives+false_negatives) 
            U = 1. * true_negatives/(2*true_negatives+false_positives) 
            auc = T + U
            s_auc = f'{auc:.3f}'
            #print(f'auc: {auc:.3f}')
            #print('*'*30)
            nom   = true_positives+true_negatives
            denom = true_positives+true_negatives+false_positives+false_negatives
            acc = 1. * (nom)/(denom)
            s_acc = f'{acc:.3f}({nom}/{denom})'
            #print(f'acc:     {s_acc}')
            nom   = true_positives
            denom = true_positives + false_positives
            pos_acc = 1. * nom / denom
            s_pos_acc = f'{pos_acc:.3f}({nom}/{denom})'
            #print(f'pos acc: {s_pos_acc}')
            nom   = true_negatives
            denom = true_negatives + false_negatives
            neg_acc = 1. * nom / denom
            s_neg_acc = f'{neg_acc:.3f}({nom}/{denom})'
            #print(f'neg acc: {neg_acc:.3f}({nom}/{denom})')
            print('*'*30)
             #['loss', 'acc', 'recall', 'precision', 'f_one', 'auc', 'positive acc', 'negative acc']
            metrics = [s_acc, s_reca, s_prec, s_fscr, s_auc, s_pos_acc, s_neg_acc, s_confusion_m]
            return metrics


if __name__ == '__main__':
    print(f'k: {args.k} q: {args.q}')
    main()