from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loss import *


# losses that need sigmoid on top of last layer
yes_softmax = ['crossentropy', 'forward', 'est_forward', 'backward',
               'est_backward', 'boot_soft', 'savage']
# unhinged needs bounded models or it diverges
yes_bound = ['unhinged', 'ramp', 'sigmoid']


class KerasModel():
    # custom losses for the CNN
    def make_loss(self, loss, P=None, binary=True):
        if loss == 'crossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if binary:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            return loss
        elif loss in ['forward', 'backward']:
            return robust(loss, P)
        elif loss == 'unhinged':
            return unhinged
        elif loss == 'sigmoid':
            return sigmoid
        elif loss == 'ramp':
            return ramp
        elif loss == 'savage':
            return savage
        elif loss == 'boot_soft':
            return boot_soft
        else:
            ValueError("Loss unknown.")
    def compile(self, loss, P=None, binary=True):
        if self.optimizer is None or self.model is None:
            print('something wrong...')
            ValueError()
        
        defined_metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')]

        self.model.compile(loss=self.make_loss(loss, P, binary), optimizer=self.optimizer, metrics=defined_metrics)
        self.model.summary()

    def direct_load_model(self, filename, epoch_resume, loss, P=None, binary=True):
        self.model = tf.keras.models.load_model(filename, custom_objects={'loss':self.make_loss(loss,P,binary)})
        self.epochs -= epoch_resume 
        print(f'Loaded model from {filename} at epoch: {epoch_resume}')

    def load_model(self, filename):
        self.model.load_weights(filename)
        print("Loaded model's weight from %s" % filename)

    def fit_model(self, model_file, epoch_resume, additional_callbacks = []):
        if self.train_loader is None or self.val_loader is None or self.epochs is None or self.batch_size is None:
            print('Parameters are not initialized correctly. ABORTING NETWORK TRANING.')
            return

        callbacks = []
        # get saver callbacks
        monitor = 'val_loss'
        best_only_saver_callback = ModelCheckpoint(f'{str(model_file)}_best_{epoch_resume}.h5', monitor=monitor, verbose=1, save_best_only=True)
        all_saver_calback = ModelCheckpoint(f'{str(model_file)}_latest.h5', monitor=monitor, verbose=1, save_best_only=False)
        # add the new callbacks to the old ones
        callbacks.append(best_only_saver_callback)
        callbacks.append(all_saver_calback)
        for c in additional_callbacks:
            callbacks.append(c)


        if hasattr(self, 'scheduler'):
            callbacks.append(self.scheduler)

        # use data augmentation
        if hasattr(self, 'train_generator'):
            print('Starting a train WITH augmentaion')
            history = \
                self.model.fit(self.train_generator, validation_data = self.validation_generator, 
                                epochs = self.epochs, batch_size=self.batch_size, verbose=1, 
                                callbacks=callbacks)

        else:        
            print('Starting a train WITHOUT augmentaion')
            history = self.model.fit(self.train_loader, validation_data=self.val_loader, 
                            epochs=self.epochs, batch_size=self.batch_size, 
                            verbose=1, callbacks=callbacks)

        # use the model that reached the lowest loss at training time
        self.load_model(model_file+'_latest.h5')
        return history.history

    def evaluate_model(self, loader):
        score = self.model.evaluate(loader, batch_size=self.num_batch, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score[1]

    def predict_proba(self, loader):
        pred = self.model.predict(loader, batch_size=self.num_batch, verbose=1)
        return pred


class ChexpertModel(KerasModel):

    def __init__(self, model, train_loader, val_loader, epochs, batch_size):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_batch = train_loader.batch_size
        self.classes = 2
        self.augmentation = True
        self.optimizer = Adam(lr=5e-4)
        #SGD(lr=0.1, momentum=0.9, decay=0.0) # TODO: ADAM ?
        # ASSIGN THE PARAMATER 'self.scheduler'
        self.lr_scheduler()
        #self.decay = 0.0001

    def prepare_data_augmentation(self):
        if self.train_loader is None or self.val_loader is None:
            print("Loader/generator does not exist. Aborting load data")
            return

        if self.augmentation:
            print('Data Augmentation')
            # data augmentation
            train_image_generator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
            val_image_generator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)

            x_train = self.train_loader.get_all_samples()
            y_train = self.train_loader.get_all_labels()
            self.train_generator = train_image_generator.flow(x_train, y_train, batch_size=self.batch_size)

            x_val = self.val_loader.get_all_samples()
            y_val = self.val_loader.get_all_labels()
            self.validation_generator = val_image_generator.flow(x_val, y_val, batch_size=self.batch_size)

    def lr_scheduler(self):
        def scheduler(epoch):
            if epoch > 80:
                return 5e-6
            elif epoch > 40:
                return 5e-5
            else:
                return 5e-4

        print('LR scheduler')
        self.scheduler = LearningRateScheduler(scheduler)

    def build_model(self, loss, P=None, binary=True):
        self.compile(loss, P, binary=binary)

class NoiseEstimator():
    def __init__(self, classifier, row_normalize=True, alpha=0.0,
                 filter_outlier=False, cliptozero=False, verbose=0):
        """classifier: an ALREADY TRAINED model. In the ideal case, classifier
        should be powerful enough to only make mistakes due to label noise."""

        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.filter_outlier = filter_outlier
        self.cliptozero = cliptozero
        self.verbose = verbose

    def fit(self, loader):
        # number of classes
        c = self.classifier.classes
        T = np.empty((c, c))
        # predict probability on the fresh sample
        eta_corr = self.classifier.predict_proba(loader)
        # find a 'perfect example' for each class
        for i in np.arange(c):
            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97,
                                           interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)
            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        return self

    def predict(self):
        T = self.T
        c = self.classifier.classes

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T
