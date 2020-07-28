import os
import sys
sys.path.append('../baseline')
sys.path.append('baseline')

import getopt
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.keras import backend as K
from noise import *
from models import *
import utils
# NECESSARY FILES FROM BASELINE FOLDER
from tf_chexpert_callbacks import *
from tf_chexpert_utilities import *
from tf_chexpert_loader import *
import densenet

def fpath(folder, loss, noise):
    return f'{folder}/forwardt/densenet121_l_{loss}_n_{noise}'

def error_and_exit():
    print('Usage: ' + str(__file__) + ' -l loss -n noise_rate -c positive_continue_epoch')
    sys.exit()

opts, args = getopt.getopt(sys.argv[1:], "l:n:c:")
noise = None
loss = None
epoch_resume = None
for opt, arg in opts:
    if opt == '-l':
        loss = arg
    elif opt == '-n':
        noise = np.array(arg).astype(np.float)
    elif opt == '-c':
        epoch_resume = int(arg)
        if epoch_resume < 0:
            error_and_exit()
    else:
        error_and_exit()

# compulsory params
if loss is None or noise is None:
    error_and_exit()

print("Params: loss=%s, noise=%s"% (loss, noise))

model = densenet.get_densenet()

train_loader, val_loader = utils.get_chexpert_loaders(float(noise), batch_size=16)

model_folder ='./models/forwardt/'
model_path = fpath('./models', loss, noise)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
# Check this.
filter_outlier = False

kerasModel = ChexpertModel(model, train_loader, val_loader, epochs=100, batch_size=16)
# TODO: maybe optimizer adam?
# kerasModel.optimizer = Adam()
P = build_uniform_P(2, noise)

if loss == 'est_forward':
    if not os.path.isfile(model_path):
        ValueError('Need to train with crossentropy first !')
    kerasModel.build_model('crossentropy', P=None, binary=True)
    kerasModel.load_model(model_path)
    # estimate P
    est = NoiseEstimator(classifier=kerasModel, alpha=0.0, filter_outlier=filter_outlier)
    # use all train_loader
    P_est = est.fit(train_loader).predict()
    print('Condition number:', np.linalg.cond(P_est))
    print('T estimated: \n', P_est)
    # compile the model with forward
    kerasModel.build_model('forward', P=P_est)
else:
    if epoch_resume == 0:
        # compile the model
        kerasModel.build_model(loss, P)
    elif epoch_resume > 0:
        # get the already saved model
        kerasModel.direct_load_model(model_path+'_latest.h5', epoch_resume, loss=loss, P=P, binary=True)
        
# some additional callbacks
prediction_save_folder = f'./network_training_predictions/forwardt_{loss}_{int(noise*100)}'
if not os.path.exists('./network_training_predictions'):
    os.mkdir('./network_training_predictions')
if not os.path.exists(prediction_save_folder):
    os.mkdir(prediction_save_folder)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
pcs = PredictionSaveCallback(train_loader, val_loader, h5_save=None, prediction_folder=prediction_save_folder, epoch_resume=epoch_resume)
#callbacks = [tensorboard_callback, pcs]
callbacks = [pcs]

#kerasModel.prepare_data_augmentation()
history = kerasModel.fit_model(model_path, epoch_resume, additional_callbacks=callbacks)
# save history
history_folder = './history/'
history_forwardt_folder = f'{history_folder}/forwardt'
if not os.path.exists(history_folder):
    os.mkdir(history_folder)
if not os.path.exists(history_forwardt_folder):
    os.mkdir(history_forwardt_folder)

history_file = fpath(history_folder, loss, noise)

# decomment for writing history
with open(history_file, 'wb') as f:
    pickle.dump(history, f)
    print('History dumped at ' + str(history_file))

# test
test_loader = get_test_loader()
score = kerasModel.evaluate_model(test_loader)
print('TEST SCORE: %s'%s)

    