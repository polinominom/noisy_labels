# -*- coding: utf-8 -*-
"""chexpert_noisfy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Up66s8R2xK4yRqBX0sGYIZpyZ1ncOnhd
"""

import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

#from google.colab import drive
#drive.mount('/content/gdrive')

# MODEL save & load
def save_model(model, destination):
  model.save(destination)

def load_model(source):
  return tf.keras.models.load_model(source)

# JSON save & load
def get_history(name):
  return json.load(open(name,"r"))

def save_history(history, name):
  json.dump(history, open(name, 'w'))

# ndarray save & load
def unpickle(fname):
    with open(fname, 'rb') as fo:
        return pickle.load(fo)
def save_ndarray(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

"""# Model Implementations

## Densenet
"""

from tensorflow.keras.applications.densenet import DenseNet121
def get_densenet():
  return DenseNet121(include_top=True, 
                      weights=None,
                      input_shape=(256, 256, 3), 
                      classes=2)

"""## Vgg"""

from tensorflow.keras.applications.vgg16 import VGG16
def get_vgg16():
  return VGG16(include_top=True, 
                weights=None,
                input_shape=(256, 256, 3), 
                classes=2)

"""## Resnet"""

from tensorflow.keras.applications.resnet import ResNet50
def get_resnet50():
  return ResNet50(include_top=True, 
                  weights=None,
                  input_shape=(256, 256, 3), 
                  classes=2)

"""## model compile func"""

def compile_model(model, binary=False):
  # Instantiate a logistic loss function that expects integer targets.
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  # Instantiate an accuracy metric.
  accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  if binary:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    accuracy = tf.keras.metrics.binary_accuracy
  
  # Instantiate an optimizer.
  optimizer = tf.keras.optimizers.Adam()
  # Instantiate some callbacks
  
  # NO EARLY STOPPING FOR NOW
  model.compile(optimizer=optimizer, loss=loss,metrics=[accuracy])
  return model

m =get_densenet()
#m.summary()

m2 = get_resnet50()
#m2.summary()

m3 = get_vgg16()
#m3.summary()

# compile model example
m = compile_model(m, binary=True)
m2 = compile_model(m2, binary=True)
m3 = compile_model(m3, binary=True)

"""# Train"""

def get_semantic_noise_generation_train_data(limit=1500):
  pos_count = limit//2
  neg_count = pos_count
  general_x = None
  general_y = None
  for i in range(1000, 131000, 1000):
    local_x = unpickle("./adjusted_data/adjusted_train_images_%i"%i)
    local_y = unpickle("./labels/train_label_%i"%i)
    if type(general_x)==type(None):
      general_x = local_x
      general_y = local_y
    else:
      general_x = np.concatenate((general_x, local_x))
      general_y = np.concatenate((general_y, local_y))

  pos_indices = np.where(general_y==1.0)[0][:pos_count]
  neg_indices = np.where(general_y!=1.0)[0][:neg_count]
  general_x = np.concatenate((general_x[pos_indices], general_x[neg_indices]))
  general_y = np.concatenate((general_y[pos_indices], general_y[neg_indices]))
  return general_x, general_y

# create history folder if it doesn't exist
if not os.path.exists("./history"):
  os.mkdir("./history")

# create models folder if it doesn't exist
if not os.path.exists("./models"):
  os.mkdir("./models")

# create semantic train folder if it doesn't exist
if not  os.path.exists("./semantic_train_data"): 
  os.mkdir("./semantic_train_data")

# Print the GPU availability
if tf.test.gpu_device_name():
  print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
  print("Unable to detect any GPU device")

# get semantic train data and their labels from the folder if they exists.
# otherwise; get the data from the adjusted folder and save it.
if os.path.exists("./semantic_train_data/sem_train") and os.path.exists("./semantic_train_data/sem_labels"):
  print("Loading semantic train data...")
  sem_x_train = unpickle("./semantic_train_data/sem_train")
  sem_y_train = unpickle("./semantic_train_data/sem_labels")
else:
  print("Semantic train data selection started...")
  sem_x_train, sem_y_train = get_semantic_noise_generation_train_data(limit=11200)
  print("Selected semantic train data saving...")
  save_ndarray("./semantic_train_data/sem_train",sem_x_train)
  save_ndarray("./semantic_train_data/sem_labels",sem_y_train)

# set number of epochs. This might change later
epochs = 200
# set batch
b=64
# convert "nan"s to "0" and binary(1-0) to categorical[1,0] in order to run the models
sem_y_train[np.where(sem_y_train!=1.0)] = 0.0
sem_y_train = tf.keras.utils.to_categorical(sem_y_train, num_classes=2)
# train and save results of densenet121
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./models/densenet', save_best_only=False), CSVLogger("./history/densenet.csv", append=True)] 
history1 = m.fit(sem_x_train, sem_y_train, epochs = epochs, batch_size = b, verbose=1, validation_split=0.8, callbacks=callbacks)
save_history(history1.history, "./history/history1.json")
save_model(m1, "./models/m1")

# train and save results of resnet50
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./models/resnet', save_best_only=False), CSVLogger("./history/resnet.csv", append=True)]
history2 = m2.fit(sem_x_train, sem_y_train, epochs = epochs, batch_size = b, verbose=1, validation_split=0.8, callbacks=callbacks)
save_history(history2.history, "./history/history2.json")
save_model(m2, "./models/m2")

# train and save results of vgg16
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./models/vgg', save_best_only=False), CSVLogger("./history/vgg.csv", append=True)]
history3 = m3.fit(sem_x_train, sem_y_train, epochs = epochs, batch_size = b, verbose=1, validation_split=0.8, callbacks=callbacks)
save_history(history3.history, "./history/history3.json")
save_model(m2, "./models/m3")

