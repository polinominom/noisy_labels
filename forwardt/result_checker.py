import os
import utils
import numpy as np
import sys
sys.path.append('baseline')
from tf_chexpert_utilities import *

def concatenate_results(p_path):
    length = len(os.listdir(p_path))//2
    print(length)
    t_gt, v_gt = utils.get_ground_truths()
    _dir = os.listdir(p_path)

    val_lst = []
    train_lst = []
    for i in range(length):
        val_fname   = f'{p_path}/val_predictions_{i}'
        train_fname = f'{p_path}/train_predictions_{i}'
        val_lst.append(val_fname)
        train_lst.append(train_fname)
    #
    train_lst = np.array(train_lst)
    val_lst = np.array(val_lst)
    #
    #print(train_lst)
    #print(val_lst)

    t_acc_lst = []
    for i, t_file in enumerate(train_lst):
        a = unpickle(t_file)
        acc = 0.0
        for j, v in enumerate(a):
            acc += np.array(np.argmax(v) == t_gt[j][1]).sum()
        acc /= len(a)
        t_acc_lst.append(acc)

    save_ndarray(f'./forwardt/forwardt_train_acc_{length}', np.array(t_acc_lst))
    print(t_acc_lst)

    v_acc_lst = []
    for i, v_file in enumerate(val_lst):
        a = unpickle(v_file)
        acc = 0.0
        for j, v in enumerate(a):
            acc += np.array(np.argmax(v) == v_gt[j][1]).sum()
        acc /= len(a)
        v_acc_lst.append(acc)

    save_ndarray(f'./forwardt/forwardt_val_acc_{length}', np.array(v_acc_lst))
    print(v_acc_lst)


p_path = '/Users/Apple/Downloads/baseline_plots/_result_forwardt_forward_20'
concatenate_results(p_path)
p_path = '/Users/Apple/Downloads/baseline_plots/_result_forwardt_est_forward_20'
concatenate_results(p_path)