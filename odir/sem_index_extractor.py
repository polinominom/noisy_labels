import os
import pandas as pd
import numpy as np
columns = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
label_dict = {}


sem_train_item_per_class = 41
sem_val_item_per_class   = 3
count_dict = {}
for c in columns:
    label_dict[c] = [] 
    count_dict[c] = sem_train_item_per_class + sem_val_item_per_class

df_all = pd.read_csv('./ground_truth/odir.csv')

total_counts = np.array(df_all['Total'])
print(f'number of the samples that have only one label:{(total_counts == 1).sum()}')
unique_samples_indices = np.where(total_counts == 1)[0]
print(unique_samples_indices)

for idx in unique_samples_indices:
    for c in columns:
        if int(df_all[c][idx]) == 1:
            if count_dict[c] > 0:
                label_dict[c].append(idx)
                count_dict[c] -= 1
        
    if np.sum(list(count_dict.values())) == 0:
        break
    
#for k,v in label_dict.items():
 #   print(f'selected indices of {k}: {v}')
  #  print(f'length: {len(v)}')

with open('./sem_indices.txt', 'w+') as fp:
    for k,values in label_dict.items():
        fp.write(f'{k}::')
        for v in values:
            fp.write(f'{v}:')
        fp.write('\n')


