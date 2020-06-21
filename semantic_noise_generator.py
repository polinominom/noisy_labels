import pandas as pd
import numpy as np
import pickle
import os

NOISE_RATIOS = [float(10*i/100) for i in range(7)]

# --- Functions
def get_flipped_by_element(element):
    return np.array([abs(element[0]-1),abs(element[1]-1)])

def get_flipped_labels_from_lst(lst):
    np.array(list(map(get_flipped_by_element, lst)))
    
def log_needed_chunks(lst,p='',t='train'):
    needed_chunk_ids = []
    for gi in lst:
        #li = int(find_local_index(gi))
        ci = int(find_chunk_id(gi))
        if not ci in needed_chunk_ids:
            needed_chunk_ids.append(ci)
    
    msg = 'needed chunk ids for %s_%s: %s'%(str(p), str(t), str(needed_chunk_ids))
    with open('./_%s_%s_chunk_list.txt'%(str(p),str(t)),'w+') as fp:
        fp.writelines(msg)
    print('length: %i'%len(needed_chunk_ids))

def get_samples_from_idx_list(lst):
    samples = []
    labels = []
    logs = []
    for gi in lst:
        li = int(find_local_index(gi))
        ci = int(find_chunk_id(gi))
        #print('gi:%i computed - ci:%i and comuted - li:%i\n'%(gi,ci,li))
        logs.append('gi:%i computed - ci:%i and comuted - li:%i\n'%(gi,ci,li))
        t_fname = './adjusted_data/adjusted_train_images_%i'%ci
        l_fname = './labels/train_label_%i'%ci
        unpickled_samples = unpickle(t_fname)
        unpickled_labels = unpickle(l_fname) 
        samples.append(unpickled_samples[li])
        labels.append(unpickled_labels[li])
        # TODO: check if this fixes memory issue
        del unpickled_samples
        del unpickled_labels
        del t_fname
        del l_name
        del li
        del ci
        

    with open('./log_file.txt', 'w+') as fp:
        fp.writelines(logs)
        del logs

    return [np.array(samples), np.array(labels)]

def find_local_index(gi):
    return gi%1000

def find_chunk_id(gi):
    return ( (gi//1000) + 1 ) * 1000
# save & load ndarray
def unpickle(fname):
    with open(fname, 'rb') as fo:
        return pickle.load(fo)
def save_ndarray(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

# x,y=get_samples_from_idx_list([100,200])
# print(x)
# exit()
# ---- ---- ---- ---- ---- ---- ----
if not os.path.exists('./noisy_data'):
    os.mkdir('./noisy_data')

# -------- -------- -------- LOADING SAMPLES -------- -------- --------
# get pos samples
train_pos_df = pd.read_csv('./selected_positive_train_samples.csv')
val_pos_df = pd.read_csv('./selected_positive_val_samples.csv')
test_pos_df = pd.read_csv('./selected_positive_test_samples.csv')
# get neg samples
train_neg_df = pd.read_csv('./selected_negative_train_samples.csv')
val_neg_df = pd.read_csv('./selected_negative_val_samples.csv')
test_neg_df = pd.read_csv('./selected_negative_test_samples.csv')
# some logs
print('length of train_pos_df:%i'%len(train_pos_df))
print('length of val_pos_df:%i'%len(val_pos_df))
print('length of test_pos_df:%i'%len(test_pos_df))
print('length of train_neg_df:%i'%len(train_neg_df))
print('length of val_neg_df:%i'%len(val_neg_df))
print('length of test_neg_df:%i'%len(test_neg_df))

print('getting samples and labels...')
# -- get train samples and labels
tp = np.array(train_pos_df['Pos:T:GlobalIndex'])
tp_sample, tp_label = get_samples_from_idx_list(tp)
tn = np.array(train_neg_df['Neg:T:GlobalIndex'])
tn_sample, tn_label = get_samples_from_idx_list(tn)
# -- get val samplesand labels
vp = np.array(val_pos_df['Pos:V:GlobalIndex'])
vp_sample, vp_label = get_samples_from_idx_list(vp)
vn = np.array(val_neg_df['Neg:V:GlobalIndex'])
vn_sample, vn_label = get_samples_from_idx_list(vn)
# -- get test samples and labels
tsp = np.array(test_pos_df['Pos:T:GlobalIndex'])
tsp_sample, tsp_label = get_samples_from_idx_list(tsp)
tsn = np.array(test_neg_df['Neg:T:GlobalIndex'])
tsn_sample, tsn_label = get_samples_from_idx_list(tsn)

print('Starting flip...')
# log_needed_chunks(tp, p='pos', t='train')
# log_needed_chunks(tn, p='neg', t='train')
# log_needed_chunks(vp, p='pos', t='val')
# log_needed_chunks(vn, p='neg', t='val')
# log_needed_chunks(tsp, p='pos', t='test')
# log_needed_chunks(tsn, p='neg', t='test')
# print('all:')
# log_needed_chunks(np.concatenate((tp,tn,vp,vn,tsp,tsn)), p='mix',t='all')
# exit()
# -------- -------- -------- NOISE SIMULATION -------- -------- --------
# simulation for all noise ratios. 'Test samples  are exclueded'
tp_flipped_labels = {}
tn_flipped_labels = {}
vp_flipped_labels = {}
vn_flipped_labels = {}
#
all_train_samples   = np.concatenate((tp_sample, tn_sample))
all_train_labes     = np.concatenate((tp_label, tn_label))
all_val_samples     = np.concatenate((vp_sample, vn_sample))
all_val_labes       = np.concatenate((vp_label, vn_label))
all_test_samples    = np.concatenate((tsp_sample, tsn_sample))
all_test_labes      = np.concatenate((tsp_label, tsn_sample))
#
shuffled_train_ids  = np.arange(len(all_train_labes))
shuffled_val_ids  = np.arange(len(all_val_labes))
shuffled_test_ids  = np.arange(len(all_test_labes))
np.random.seed(1000)
np.random.shuffle(shuffled_train_ids)
np.random.seed(1001)
np.random.shuffle(shuffled_val_ids)
np.random.seed(1002)
np.random.shuffle(shuffled_test_ids)
print('Simulation ALL noises:')
for n in NOISE_RATIOS:
    print('Starting to simulate noisy samples for [%.1f%s] percentage...'%(n*100,'%'))
    print('total noisy sample count in train_pos_df:%.1f'%len(tp_sample)*n)
    print('total noisy sample count val_pos_df:%.1f'%len(vp_sample)*n)
    print('total noisy sample count train_neg_df:%.1f'%len(tn_sample)*n)
    print('total noisy sample count val_neg_df:%.1f'%len(vn_sample)*n)
    # flip train pos
    c = len(tp_sample)*n
    fl = get_flipped_labels_from_lst(tp_label[:c]) 
    tp_flipped_labels[n] = np.concatenate((fl,tp_label[c:]))
    # flip val pos
    c = len(vp_sample)*n
    fl = get_flipped_labels_from_lst(vp_label[:c]) 
    vp_flipped_labels[n] = np.concatenate((fl,vp_label[c:]))
    # flip train neg
    c = len(tn_sample)*n
    fl = get_flipped_labels_from_lst(tn_label[:c]) 
    tn_flipped_labels[n] = np.concatenate((fl,tn_label[c:]))
    # flip val neg
    c = len(vn_sample)*n
    fl = get_flipped_labels_from_lst(vn_label[:c]) 
    vn_flipped_labels[n] = np.concatenate((fl,vn_label[c:]))
    
    # SAVE THE FILE
    # EXAMPLE: fnames for 20 percent noisy files
    # ---- 20_train_set.pkl
    # ---- 20_val_set.pkl
    # ---- test_set.pkl
    fname = '%i_train_set.pkl'%(int(n*100))
    print('saving %s...'%fname)
    all_train_flipped = np.concatenate((tp_flipped_labels[n], tn_flipped_labels[n]))
    x = all_train_samples[shuffled_train_ids]
    y = all_train_labes[shuffled_train_ids]
    z = all_train_flipped[shuffled_train_ids]
    s_data = np.array([x, y, z])
    save_ndarray('./noisy_data/%s'%fname, s_data)
    # val
    fname = '%i_val_set.pkl'%(int(n*100))
    print('saving %s...'%fname)
    all_val_flipped = np.concatenate((vp_flipped_labels[n], vn_flipped_labels[n]))
    x = all_val_samples[shuffled_val_ids]
    y = all_val_labes[shuffled_val_ids]
    z = all_val_flipped[shuffled_val_ids]
    s_data = np.array([x, y, z])
    save_ndarray('./noisy_data/%s'%fname, s_data)

# test
fname = 'test_set.pkl'
print('saving %s...'%fname)
x = all_test_samples[shuffled_test_ids]
y = all_test_labes[shuffled_test_ids]
s_data = np.array([x, y])
save_ndarray('./noisy_data/%s'%fname, s_data)