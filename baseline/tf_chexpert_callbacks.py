import sys
sys.path.append('./baseline')
from tf_chexpert_utilities import *
import tensorflow as tf
import numpy as np
import h5py

# define the custom callback for prediction save
class PredictionSaveCallback(tf.keras.callbacks.Callback):
  def __init__(self, train_loader, validation_loader, h5_save, prediction_folder, epoch_resume):
    super(PredictionSaveCallback, self).__init__()
    self.h5_save = h5_save
    self.train_loader = train_loader
    self.val_loader = validation_loader
    self.epoch_resume = epoch_resume
    self.epoch=None
    self.batch_size=self.train_loader.batch_size
    self.train_predictions = np.zeros((self.train_loader.get_total_item_count(), 2)) 
    self.val_predictions = np.zeros((self.val_loader.get_total_item_count(), 2))
    self.prediction_folder = prediction_folder
    #self.task_queue = TaskQueue(num_workers=1)

  def on_epoch_begin(self, epoch, logs={}):
    self.epoch=epoch

  # to get train predictions
  def on_train_batch_end(self, batch, logs={}):
    x, y = self.train_loader[batch]
    y_pred_t = self.model.predict(x)
    for idx in range(len(y_pred_t)):
      item_id = batch * len(y_pred_t) + idx
      self.train_predictions[item_id] = y_pred_t[idx]

  # to get validation predictions
  def on_test_batch_end(self, batch, logs={}):
    x, y = self.val_loader[batch]
    y_pred_v = self.model.predict(x)
    for idx in range(len(y_pred_v)):
      item_id = batch * len(y_pred_v) + idx
      self.val_predictions[item_id] = y_pred_v[idx]
  
  # TO SAVE ALL TRAIN AND VALIDATION PREDICTONS
  def on_epoch_end(self, epoch, logs={}):
    e = int(epoch) + self.epoch_resume
    #save train predictions
    fname = f'{self.prediction_folder}/train_predictions_{e}'
    save_ndarray(fname, self.train_predictions)
    #save val predictions
    fname = f'{self.prediction_folder}/val_predictions_{e}'
    save_ndarray(fname, self.val_predictions)
    # reset
    self.train_predictions = self.train_predictions * 0
    self.val_predictions = self.val_predictions * 0
    
# TAKEN FROM: https://www.tensorflow.org/guide/keras/custom_callback#early_stopping_at_minimum_loss
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  def __init__(self, model_save_dir, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()
    self.patience = patience
    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None
    self.model_save_dir = model_save_dir

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get("val_loss")
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print("Restoring model weights from the end of the best epoch.")
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      fname = '%s/%i'%(self.model_save_dir, self.stopped_epoch + 1)
      print("Epoch %05d: early stopping. Saving best weights to: %s" % (self.stopped_epoch + 1, fname))
      self.model.save_weights(fname)