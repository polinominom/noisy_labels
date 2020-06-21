import pandas as pd
import numpy as np

sorted_df = pd.read_csv('./reverse_sorted.csv')
#print(sorted_df.head())
# ground truths
gt_df = sorted_df["ground_truth"]
gt = np.array(gt_df)
# get positives
pos_relative_ids = np.where(gt=='[0.0, 1.0]')[0]
#print('pos_relative_ids: %s'%str(pos_relative_ids))
pos_global_ids = np.array(sorted_df["global_index"])[pos_relative_ids]
pos_losses = np.array(sorted_df["loss"])[pos_relative_ids]
# get negatives
neg_relative_ids = np.where(gt=='[1.0, 0.0]')[0]
neg_global_ids = np.array(sorted_df["global_index"])[neg_relative_ids]
neg_losses = np.array(sorted_df["loss"])[neg_relative_ids]
# get min count
print('healthy sample count: %i'%len(pos_global_ids))
print('not-healthy sample count: %i'%len(neg_global_ids))
min_sample_count = min(len(pos_global_ids), len(neg_global_ids))
if min_sample_count % 10 != 0:
    print("ERROR: Sample count is indivisible to 10(%i). Need to provide a new method, noise generation stopped!"%min_sample_count)
# Get samples with 8 - 1 - 1 from

limit=min_sample_count//10
print('limit:%i'%limit)

train_pos = np.zeros(limit*8)
train_neg = np.zeros(limit*8)
val_pos   = np.zeros(limit)
val_neg   = np.zeros(limit)
test_pos  = np.zeros(limit)
test_neg  = np.zeros(limit)
train_loss_pos  = np.zeros(limit*8)
train_loss_neg  = np.zeros(limit*8)
val_loss_pos    = np.zeros(limit)
val_loss_neg    = np.zeros(limit)
test_loss_pos   = np.zeros(limit)
test_loss_neg   = np.zeros(limit)

if limit == 0:
    print('0 samples has been selected')
    exit()
for idx in range(limit):
    i = 10*idx
    print('i:%i'%i)
    # --- pos - train
    ti = 8*idx
    train_pos[ti:ti+8]      = pos_global_ids[i:i+8]
    train_loss_pos[ti:ti+8] = pos_losses[i:i+8]
    # --- pos - val
    val_pos[idx]            = pos_global_ids[i+8]
    val_loss_pos[idx]       = pos_losses[i+8]
    # --- pos - test
    test_pos[idx]           = pos_global_ids[i+9]
    test_loss_pos[idx]      = pos_losses[i+9]
    # --- neg - train
    train_neg[ti:ti+8]      = neg_global_ids[i:i+8]
    train_loss_neg[ti:ti+8] = neg_losses[i:i+8]
    # --- neg - val
    val_neg[idx]            = neg_global_ids[i+8]
    val_loss_neg[idx]       = neg_losses[i+8]
    # --- neg - test
    test_neg[idx]           = neg_global_ids[i+9]
    test_loss_neg[idx]      = neg_losses[i+9]


# SAVE CSV
# --- pos - train
n1 = 'Pos:T:GlobalIndex'
n2 = 'Pos:T:Loss'
data = {n1:train_pos, n2:train_loss_pos}
train_pos_df = pd.DataFrame(data, columns=[n1, n2])
train_pos_df.to_csv('./selected_positive_train_samples.csv', index=False)
# --- pos - val
n1 = 'Pos:V:GlobalIndex'
n2 = 'Pos:V:Loss'
data = {n1:val_pos, n2:val_loss_pos}
train_pos_df = pd.DataFrame(data, columns=[n1, n2])
train_pos_df.to_csv('./selected_positive_val_samples.csv', index=False)
# --- pos - test
n1 = 'Pos:T:GlobalIndex'
n2 = 'Pos:T:Loss'
data = {n1:test_pos, n2:test_loss_pos}
train_pos_df = pd.DataFrame(data, columns=[n1, n2])
train_pos_df.to_csv('./selected_positive_test_samples.csv', index=False)
# --- neg - train
n1 = 'Neg:T:GlobalIndex'
n2 = 'Neg:T:Loss'
data = {n1:train_neg, n2:train_loss_neg}
train_pos_df = pd.DataFrame(data, columns=[n1, n2])
train_pos_df.to_csv('./selected_negative_train_samples.csv', index=False)
# --- neg - val
n1 = 'Neg:V:GlobalIndex'
n2 = 'Neg:V:Loss'
data = {n1:val_neg, n2:val_loss_neg}
train_pos_df = pd.DataFrame(data, columns=[n1, n2])
train_pos_df.to_csv('./selected_negative_val_samples.csv', index=False)
# --- neg - test
n1 = 'Neg:T:GlobalIndex'
n2 = 'Neg:T:Loss'
data = {n1:test_neg, n2:test_loss_neg}
train_pos_df = pd.DataFrame(data, columns=[n1, n2])
train_pos_df.to_csv('./selected_negative_test_samples.csv', index=False)