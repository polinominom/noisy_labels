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
import tf_chexpert_utilities, tf_chexpert_callbacks, tf_chexpert_loader
import densenet

def fpath(folder, loss, noise):
    return '%s/forwardt/densenet121_l_%s_n_%s'%(folder, str(loss),str(noise))

def error_and_exit():
    print('Usage: ' + str(__file__) + ' -l loss ' + '-n noise_rate')
    sys.exit()

opts, args = getopt.getopt(sys.argv[1:], "l:n:")
noise = None
loss = None
for opt, arg in opts:
    if opt == '-l':
        loss = arg
    elif opt == '-n':
        noise = np.array(arg).astype(np.float)
    else:
        error_and_exit()

# compulsory params
if loss is None or noise is None:
    error_and_exit()

print("Params: loss=%s, noise=%s"% (loss, noise))
    
model = densenet.get_densenet()
train_loader, val_loader = utils.get_chexpert_loaders(float(noise), batch_size=32)

model_folder ='./models/forwardt/'
model_path = fpath('./models', loss, noise)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
# Check this.
filter_outlier = False

kerasModel = ChexpertModel(model, train_loader, val_loader)
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
    # compile the model
    kerasModel.build_model(loss, P)
        
# some additional callbacks
prediction_save_folder = './network_training_predictions/forwardt'
if not os.path.exists('./network_training_predictions'):
    os.mkdir('./network_training_predictions')
if not os.path.exists(prediction_save_folder):
    os.mkdir(prediction_save_folder)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
pcs = PredictionSaveCallback(train_loader, val_loader, prediction_save_folder='%s/%i'%(prediction_save_folder, int(100*opt.noise_ratio)))
callbacks = [tensorboard_callback, pcs]

history = kerasModel.fit_model(model_path, additional_callbacks=callbacks)
history_fname = './history/'
history_file = fpath('./history', loss, noise)

# decomment for writing history
with open(history_file, 'wb') as f:
    pickle.dump(history, f)
    print('History dumped at ' + str(history_file))

# test
test_loader = get_test_loader()
score = kerasModel.evaluate_model(test_loader)
print('TEST SCORE: %s'%s)

    