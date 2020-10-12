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

parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
#parser.add_argument('--dataset', default='covid', type=str)
parser.add_argument('--decay', default=1e-4, type=float,help='weight decay (default=1e-4)')
parser.add_argument('--lr', default=1e-4, type=float,help='initial learning rate')
parser.add_argument('--batch_size', '-b', default=10,type=int, help='mini-batch size (default: 10)')
parser.add_argument('--epochs', default=120, type=int,help='number of total epochs to run')
parser.add_argument('--start_prune', default=40, type=int,help='number of total epochs to run')
#_parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--gamma', type = float, default = 0.1)
parser.add_argument('--schedule', nargs='+', type=int)
#parser.add_argument('--k', default=0.5, type=float, help='hyperparameter for the custom loss')
#parser.add_argument('--q', default=0.7, type=float, help='hyperparameter for the custom loss')

best_acc = 0
args = parser.parse_args()

checkpoint_folder = 'covid_gce_checkpoint'
if not os.path.isdir(checkpoint_folder):
    os.mkdir(checkpoint_folder)

result_folder = './covid_gce_results'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def main(k, q):
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

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(checkpoint_folder), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./{checkpoint_folder}/current_net_n_{int(args.noise_rate*100)}_k_{int(k*100)}_q_{int(q*100)}.h5')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model.. (Default : Densenet121)')
        start_epoch = 0
        net = get_densenet()

    logname =  f'{result_folder}/covid_gce_n_{int(args.noise_rate*100)}_k_{int(k*100)}_q_{int(q*100)}.csv'

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = TruncatedLoss(trainset_size=len(trainset), q=q, k=k).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([
                'epoch', 'train loss', 'train acc', 'train real loss',
                'train real acc', 'train recall', 'train precision',
                'train auc', 'train f_one','val loss', 'val real loss', 
                'val acc', 'val real acc', 'val recall', 'val precision',
                'val auc', 'val f_one'])

    for epoch in range(start_epoch, args.epochs):
        # metrics = (loss, acc, real_loss, real_acc, recall, prec, auc, f_one)
        train_metrics = train(epoch, trainloader, net, criterion, optimizer)
        val_metrics   = val(epoch, valloader, net, criterion, k=k, q=q)
        # concatenate metrics
        results = np.concatenate((train_metrics, val_metrics))
        # insert epoch to the beginning
        results = np.insert(results, 0, epoch, axis=0)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(results)
        scheduler.step()

# Training
def train(epoch, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_real_loss = 0
    train_real_acc = 0
    correct = 0
    total = 0
    train_recall = 0
    train_auc = 0
    train_precision = 0
    train_f_one_score = 0
    #if (epoch+1) >= args.start_prune and (epoch+1) % 10 == 0:
     #   checkpoint = torch.load(f'./{checkpoint_folder}/best_ckpt.h5')
     #   net = checkpoint['net']
     #   net.eval() 
     #   for batch_idx, (inputs, targets, real_ground_truths, indexes) in enumerate(trainloader):
     #       inputs, targets = inputs.cuda(), targets.cuda()
     #       outputs = net(inputs)
     #       criterion.update_weight(outputs, targets, indexes)

        # now = torch.load(f'./{checkpoint_folder}/current_net.h5')
        # net = now['current_net']
        # net.train()

    for batch_idx, data in enumerate(trainloader):
        inputs, targets, real_ground_truths, indexes = data
        #inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        real_ground_truths = real_ground_truths.cuda()
        indexes = indexes.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        # get real loss and real acc
        train_real_loss += criterion(outputs, real_ground_truths, indexes).data.cpu()
        train_real_acc  += predicted.eq(real_ground_truths.data).cpu().sum()
        # get recall-f_1_score-auc-precision
        softmax_pred = F.softmax(outputs, dim=1).cpu().data
        train_recall      += recall_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)
        train_f_one_score += f1_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)
        train_precision   += precision_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)
        train_auc         += roc_auc_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)

        str_loss = f'loss: {train_loss / (batch_idx + 1):.3f}'
        str_acc = f'acc: {(100. * correct / total):.3f}%({correct:d}{total:d})'
        str_real_loss = f'real loss: {train_real_loss / (batch_idx + 1):.3f}'
        str_real_acc = f'real acc: {(100. * train_real_acc / total):.3f}%({train_real_acc:d}{total:d})'
        progress_bar(batch_idx, len(trainloader), f'{str_loss} | {str_acc} | {str_real_loss} | {str_real_acc}')
    # get metrics
    loss        = train_loss / len(trainloader)
    real_loss   = train_real_loss / len(trainloader)
    acc         = 100. * correct / total
    real_acc    = float(train_real_acc)/float(total)
    
    recall  = train_recall/len(trainloader)
    prec    = train_precision/len(trainloader)
    auc     = train_auc/len(trainloader)
    f_one   = train_f_one_score/len(trainloader)

    metrics = (loss, acc, real_loss, real_acc, recall, prec, auc, f_one)
    return metrics


def val(epoch, valloader, net, criterion, k, q):
    global best_acc
    net.eval()

    val_loss = 0
    val_real_loss = 0
    val_real_acc = 0
    correct = 0
    total = 0
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

            outputs = net(inputs)
            loss = criterion(outputs, targets,indexes)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            # get real loss and real acc
            val_real_loss += criterion(outputs, real_ground_truths, indexes).data.cpu()
            val_real_acc  += predicted.eq(real_ground_truths.data).cpu().sum()
            # get recall-f_1_score-auc-precision
            softmax_pred = F.softmax(outputs, dim=1).cpu().data
            val_recall      += recall_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)
            val_f_one_score += f1_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)
            val_precision   += precision_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)
            val_auc         += roc_auc_score(real_ground_truths.cpu(), softmax_pred.cpu().data[:,1] > 0.5)

            str_loss = f'loss: {val_loss / (batch_idx + 1):.3f}'
            str_acc = f'acc: {(100. * correct / total):.3f}%({correct:d}{total:d})'
            str_real_loss = f'real loss: {val_real_loss / (batch_idx + 1):.3f}'
            str_real_acc = f'real acc: {(100. * val_real_acc / total):.3f}%({val_real_acc:d}{total:d})'
            progress_bar(batch_idx, len(valloader), f'{str_loss} | {str_acc} | {str_real_loss} | {str_real_acc}')

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch, net)

    state = {
        'current_net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, f'./{checkpoint_folder}/current_net_n_{int(args.noise_rate)}_k_{int(k*100)}_q_{int(q*100)}.h5')


    # get metrics
    loss        = val_loss / len(valloader)
    real_loss   = val_real_loss / len(valloader)
    acc         = 100. * correct / total
    real_acc    = float(val_real_acc) / float(total)
    #
    recall  = val_recall/len(valloader)
    prec    = val_precision/len(valloader)
    auc     = val_auc/len(valloader)
    f_one   = val_f_one_score/len(valloader)

    metrics = (loss, acc, real_loss, real_acc, recall, prec, auc, f_one)
    return metrics


def checkpoint(acc, epoch, net):
    # Save checkpoint.
    print('Saving best....')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    
    checkpoint_folder = 'covid_gce_checkpoint'
    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    
    torch.save(state, f'./{checkpoint_folder}/best_ckpt_n_{int(args.noise_rate*100)}_k_{int(k*100)}_q_{int(q*100)}.h5')


if __name__ == '__main__':
    k_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for k in k_list:
        for q in q_list:
            print(':'*20+f'K:{k}|||q:{q}' +':'*20)
            main(k=k, q=q)